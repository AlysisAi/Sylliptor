from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pytest
from rich.console import Console

import sylliptor_agent_cli.agent_loop as agent_loop_mod
import sylliptor_agent_cli.verify_gate as verify_gate_mod
from sylliptor_agent_cli.agent_loop import AgentRuntimeError, build_tools, create_session
from sylliptor_agent_cli.config import AppConfig, clone_cfg
from sylliptor_agent_cli.runtime_kind import RuntimeKind
from sylliptor_agent_cli.session_store import SessionStore, read_session_events
from sylliptor_agent_cli.surface.noop_surface import NoopSurface
from sylliptor_agent_cli.verify_gate import ResolvedVerifyCommands, VerifyRunResult


def _store(root: Path, *, session_id: str = "verify-test") -> SessionStore:
    return SessionStore(
        enabled=False,
        sessions_dir=root / "sessions",
        session_id=session_id,
        cwd=str(root),
        repo_root=str(root),
    )


def _cp(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args="cmd",
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _host_verify_cfg(cfg: AppConfig | None = None) -> AppConfig:
    effective = clone_cfg(cfg or AppConfig(model="test-model"))
    extra_fields = dict(effective.extra_fields)
    verify_sandbox = dict(extra_fields.get("verify_sandbox") or {})
    verify_sandbox.setdefault("mode", "off")
    extra_fields["verify_sandbox"] = verify_sandbox
    effective.extra_fields = extra_fields
    return effective


def _build_tools(
    tmp_path: Path,
    *,
    mode: str = "auto",
    cfg: AppConfig | None = None,
    surface: NoopSurface | None = None,
    yes: bool = True,
    non_interactive: bool = True,
    verification_enabled: bool = True,
    authoritative_verification_commands: list[str] | None = None,
    effective_verification_commands: list[str] | None = None,
    verify_command_selection: ResolvedVerifyCommands | None = None,
) -> dict[str, object]:
    return build_tools(
        root=tmp_path,
        console=Console(file=io.StringIO(), force_terminal=False),
        surface=surface,
        store=_store(tmp_path),
        mode=mode,
        yes=yes,
        cfg=_host_verify_cfg(cfg),
        non_interactive=non_interactive,
        verification_enabled=verification_enabled,
        authoritative_verification_commands=authoritative_verification_commands,
        effective_verification_commands=effective_verification_commands,
        verify_command_selection=verify_command_selection,
    )


def _write_repo_files(root: Path, files: dict[str, str]) -> None:
    for relpath, body in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def _create_interactive_session(
    repo: Path,
    *,
    sessions_dir: Path,
    session_id: str,
    cfg: AppConfig | None = None,
    verify_cmd: list[str] | None = None,
):
    return create_session(
        cfg=_host_verify_cfg(cfg),
        root=repo,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=False,
        api_key_override="override-key",
        runtime_kind=RuntimeKind.INTERACTIVE_CHAT,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
        verify_cmd=verify_cmd,
    )


def test_build_tools_registers_verify_run(tmp_path: Path) -> None:
    tools = _build_tools(tmp_path)

    assert "verify_run" in tools
    schema = tools["verify_run"].as_openai_tool()["function"]["parameters"]
    assert schema["required"] == []
    assert schema["properties"]["commands"]["type"] == "array"


def test_build_tools_can_disable_verify_run(tmp_path: Path) -> None:
    tools = _build_tools(tmp_path, verification_enabled=False)

    assert "verify_run" not in tools


def test_verify_run_uses_configured_commands_and_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        if cmd == "pytest -q":
            return _cp(returncode=0, stdout="tests ok\n")
        return _cp(returncode=1, stderr="lint failed\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q", "ruff check ."]
    tools = _build_tools(tmp_path, cfg=cfg)

    result = tools["verify_run"].run({})

    assert calls == ["pytest -q", "ruff check ."]
    assert result["commands"] == ["pytest -q", "ruff check ."]
    assert result["all_passed"] is False
    assert result["failed_commands"] == ["ruff check ."]
    assert result["summary"] == "verification failed (1/2); failed: ruff check ."
    assert result["fallback_used"] is False
    assert result["fallback_count"] == 0
    assert result["fallback_details"] == []
    assert result["primary_failure"] == {
        "command": "ruff check .",
        "effective_command": "ruff check .",
        "snippet": "lint failed",
        "output_truncated": False,
        "fallback_used": False,
    }
    command_results = result["command_results"]
    assert command_results[0] == {
        "command": "pytest -q",
        "effective_command": "pytest -q",
        "exit_code": 0,
        "status": "passed",
        "ok": True,
        "real_execution": True,
        "output_preview": "tests ok\n",
        "output_chars": 9,
        "output_truncated": False,
        "fallback_used": False,
    }
    failed_command = dict(command_results[1])
    failure_summary = failed_command.pop("failure_summary")
    assert failed_command == {
        "command": "ruff check .",
        "effective_command": "ruff check .",
        "exit_code": 1,
        "status": "failed",
        "ok": False,
        "real_execution": None,
        "output_preview": "lint failed\n",
        "output_chars": 12,
        "output_truncated": False,
        "fallback_used": False,
    }
    assert failure_summary["primary_error"] == "lint failed"

    artifact_rel = result["artifact_path"]
    assert artifact_rel == "sessions/verify-test/verify/step001_verify_run.txt"
    assert result["artifact_saved"] is True
    assert result["artifact_readable_via_fs"] is True
    assert result["artifact_location"] == "workspace_root"
    artifact = tmp_path / artifact_rel
    assert artifact.exists() is True
    body = artifact.read_text(encoding="utf-8")
    assert "## Command 1" in body
    assert "## Command 2" in body
    assert "lint failed" in body


def test_verify_run_treats_pytest_exit_5_no_tests_as_skipped_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        assert cmd == "pytest -q"
        (tmp_path / ".pytest_cache").mkdir()
        return _cp(returncode=5, stdout="custom plugin suppressed the collection summary\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q"]
    tools = _build_tools(tmp_path, cfg=cfg)

    result = tools["verify_run"].run({})

    assert result["all_passed"] is True
    assert result["failed_commands"] == []
    assert result["summary"] == "verification skipped: nothing to verify (1/1)"
    assert result["command_results"][0]["status"] == "skipped"
    assert result["command_results"][0]["ok"] is True
    assert result["command_results"][0]["real_execution"] is False
    assert result["command_results"][0]["non_execution_reason"] == "pytest_no_tests_collected"
    assert not (tmp_path / ".pytest_cache").exists()


def test_verify_run_respects_session_log_dir_override_outside_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(_cmd, **_kwargs):  # type: ignore[no-untyped-def]
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    sessions_dir = tmp_path / "runtime" / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model"),
        root=repo,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="verify-override",
    )
    try:
        result = session.tools["verify_run"].run({"commands": ["pytest -q"]})
    finally:
        session.close()

    expected = sessions_dir / "verify-override" / "verify" / "step001_verify_run.txt"
    assert result["artifact_path"] is None
    assert result["artifact_saved"] is True
    assert result["artifact_readable_via_fs"] is False
    assert result["artifact_location"] == "external_session_store"
    assert expected.exists() is True


def test_verify_run_logs_real_external_artifact_path_in_session_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(_cmd, **_kwargs):  # type: ignore[no-untyped-def]
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    sessions_dir = tmp_path / "runtime" / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model"),
        root=repo,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="verify-event-log",
    )
    try:
        result = session.tools["verify_run"].run({"commands": ["pytest -q"]})
    finally:
        session.close()

    expected = sessions_dir / "verify-event-log" / "verify" / "step001_verify_run.txt"
    events = list(read_session_events(sessions_dir / "verify-event-log.jsonl"))
    verify_events = [event for event in events if event.get("type") == "verify_run"]

    assert result["artifact_path"] is None
    assert verify_events
    assert verify_events[-1]["payload"]["artifact_path"] == os.fspath(expected.resolve())
    assert verify_events[-1]["payload"]["model_artifact_path"] is None
    assert verify_events[-1]["payload"]["artifact_saved"] is True
    assert verify_events[-1]["payload"]["artifact_readable_via_fs"] is False
    assert verify_events[-1]["payload"]["artifact_location"] == "external_session_store"
    assert verify_events[-1]["payload"]["fallback_used"] is False
    assert verify_events[-1]["payload"]["fallback_count"] == 0
    assert verify_events[-1]["payload"]["fallback_details"] == []


def test_interactive_docs_only_pathless_prompt_updates_verification_contract_without_pytest_fallback(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id="interactive-docs-verify",
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction="Fix the bug.",
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == []
        assert session.verification_selection_source == "repo_scan.no_authoritative_commands"
        assert (
            session.verification_selection_reason
            == "repo scan found a docs-only workspace with no authoritative verification surface"
        )
        assert session.verification_contract_type == "unavailable"
        assert session.verification_authoritative is False

        env_message = next(
            (
                str(message.get("content") or "")
                for message in session.messages
                if message.get("role") == "user"
                and "<environment_context>" in str(message.get("content") or "")
            ),
            "",
        )
        assert 'verification_selection_source: "repo_scan.no_authoritative_commands"' in env_message
        assert "verification_authoritative: false" in env_message

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert result["commands"] == []
    assert result["summary"] == "verification skipped: no commands"
    assert result["all_passed"] is True

    events = list(read_session_events(sessions_dir / "interactive-docs-verify.jsonl"))
    contract_updates = [
        event for event in events if event.get("type") == "verification_contract_updated"
    ]
    verify_events = [event for event in events if event.get("type") == "verify_run"]

    assert contract_updates
    assert contract_updates[-1]["payload"]["verification_selection_source"] == (
        "repo_scan.no_authoritative_commands"
    )
    assert contract_updates[-1]["payload"]["verification_contract_type"] == "unavailable"
    assert contract_updates[-1]["payload"]["verification_authoritative"] is False
    assert verify_events
    assert verify_events[-1]["payload"]["commands"] == []
    assert verify_events[-1]["payload"]["verification_selection_source"] == (
        "repo_scan.no_authoritative_commands"
    )
    assert verify_events[-1]["payload"]["verification_authoritative"] is False


@pytest.mark.parametrize(
    ("session_id", "files", "instruction", "expected_commands", "expected_reason"),
    [
        (
            "interactive-js-auth-verify",
            {
                "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
                "src/index.ts": "export const value = 1;\n",
            },
            "Update the auth flow.",
            ["npm run build"],
            "repo scan discovered package.json verification scripts",
        ),
        (
            "interactive-js-bug-verify",
            {
                "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
                "src/index.ts": "export const value = 1;\n",
            },
            "Fix the bug.",
            ["npm run build"],
            "repo scan discovered package.json verification scripts",
        ),
        (
            "interactive-ci-verify",
            {
                ".github/workflows/ci.yml": (
                    "name: ci\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
                )
            },
            "Fix the bug.",
            [],
            "repo scan found a CI-only workspace with no authoritative verification surface",
        ),
        (
            "interactive-terraform-verify",
            {"versions.tf": ('terraform {\n  required_version = ">= 1.6.0"\n}\n')},
            "Adjust the policy.",
            [],
            "repo scan found a Terraform-only workspace with no authoritative verification surface",
        ),
        (
            "interactive-compose-verify",
            {"compose.yaml": ("services:\n  web:\n    image: nginx:latest\n")},
            "Adjust startup order.",
            [],
            "repo scan found a Compose-only workspace with no authoritative verification surface",
        ),
    ],
)
def test_interactive_pathless_non_python_tasks_do_not_inherit_generic_pytest_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_id: str,
    files: dict[str, str],
    instruction: str,
    expected_commands: list[str],
    expected_reason: str,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(repo, files)
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id=session_id,
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction=instruction,
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == expected_commands
        assert session.verification_selection_source == (
            "repo_scan.likely_test_commands"
            if expected_commands
            else "repo_scan.no_authoritative_commands"
        )
        assert session.verification_selection_reason == expected_reason
        assert session.verification_contract_type == (
            "repo_native" if expected_commands else "unavailable"
        )
        assert session.verification_authoritative is bool(expected_commands)

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert [call for call in calls if call in expected_commands] == expected_commands
    assert result["commands"] == expected_commands
    assert result["all_passed"] is True

    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    contract_updates = [
        event for event in events if event.get("type") == "verification_contract_updated"
    ]
    assert contract_updates
    assert contract_updates[-1]["payload"]["verification_selection_reason"] == expected_reason
    assert contract_updates[-1]["payload"]["verification_contract_type"] == (
        "repo_native" if expected_commands else "unavailable"
    )
    assert contract_updates[-1]["payload"]["verification_authoritative"] is bool(expected_commands)


@pytest.mark.parametrize(
    (
        "session_id",
        "files",
        "instruction",
        "expected_path",
        "expected_commands",
        "expected_reason",
    ),
    [
        (
            "interactive-js-env-example-verify",
            {
                "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
                ".env.example": "API_URL=\n",
            },
            "Update .env.example handling.",
            ".env.example",
            ["npm run build"],
            "repo scan discovered package.json verification scripts",
        ),
        (
            "interactive-js-npmrc-verify",
            {
                "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
                ".npmrc": "save-exact=true\n",
            },
            "Adjust .npmrc defaults.",
            ".npmrc",
            ["npm run build"],
            "repo scan discovered package.json verification scripts",
        ),
        (
            "interactive-js-vercel-verify",
            {
                "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
                "vercel.json": '{\n  "framework": "vite"\n}\n',
            },
            "Update vercel.json config.",
            "vercel.json",
            ["npm run build"],
            "repo scan discovered package.json verification scripts",
        ),
        (
            "interactive-python-env-example-verify",
            {
                "pyproject.toml": '[project]\nname = "demo"\nversion = "0.1.0"\n',
                ".env.example": "UPLOAD_DIR=\n",
                "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
            },
            "Update .env.example defaults.",
            ".env.example",
            [],
            "repo scan found a Python workspace without a discoverable test surface",
        ),
    ],
)
def test_interactive_neutral_config_paths_keep_repo_grounded_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_id: str,
    files: dict[str, str],
    instruction: str,
    expected_path: str,
    expected_commands: list[str],
    expected_reason: str,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(repo, files)
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id=session_id,
        cfg=AppConfig(model="test-model", verify_commands=["pytest -q"]),
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction=instruction,
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == expected_commands
        assert session.verification_selection_source == (
            "repo_scan.likely_test_commands"
            if expected_commands
            else "repo_scan.no_authoritative_commands"
        )
        assert session.verification_selection_reason == expected_reason
        assert session.verification_contract_type == (
            "repo_native" if expected_commands else "unavailable"
        )
        assert session.verification_authoritative is bool(expected_commands)

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert [call for call in calls if call in expected_commands] == expected_commands
    assert result["commands"] == expected_commands
    assert result["all_passed"] is True

    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    contract_updates = [
        event for event in events if event.get("type") == "verification_contract_updated"
    ]
    assert contract_updates
    assert contract_updates[-1]["payload"]["verification_selection_source"] == (
        "repo_scan.likely_test_commands"
        if expected_commands
        else "repo_scan.no_authoritative_commands"
    )
    assert contract_updates[-1]["payload"]["verification_selection_reason"] == expected_reason
    assert contract_updates[-1]["payload"]["instruction_paths"] == [expected_path]


@pytest.mark.parametrize(
    ("session_id", "instruction"),
    [
        ("interactive-mixed-bug-verify", "Fix the bug."),
        ("interactive-mixed-auth-verify", "Update the auth flow."),
        ("interactive-mixed-parser-verify", "Refactor the parser."),
    ],
)
def test_interactive_mixed_workspace_vague_prompts_use_repo_grounded_invalidation(
    tmp_path: Path,
    session_id: str,
    instruction: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(
        repo,
        {
            "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
            "pyproject.toml": '[project]\nname = "demo"\nversion = "0.1.0"\n',
            "src/index.ts": "export const value = 1;\n",
            "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
        },
    )
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id=session_id,
        cfg=AppConfig(model="test-model", verify_commands=["pytest -q"]),
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction=instruction,
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == []
        assert session.verification_selection_source == "repo_scan.no_authoritative_commands"
        assert (
            session.verification_selection_reason
            == "repo scan found a mixed workspace without an authoritative verification surface"
        )
        assert session.verification_contract_type == "unavailable"
        assert session.verification_authoritative is False

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert result["commands"] == []
    assert result["summary"] == "verification skipped: no commands"
    assert result["all_passed"] is True

    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    contract_updates = [
        event for event in events if event.get("type") == "verification_contract_updated"
    ]
    assert contract_updates
    assert contract_updates[-1]["payload"]["verification_selection_source"] == (
        "repo_scan.no_authoritative_commands"
    )
    assert (
        contract_updates[-1]["payload"]["verification_selection_reason"]
        == "repo scan found a mixed workspace without an authoritative verification surface"
    )
    assert contract_updates[-1]["payload"]["verification_contract_type"] == "unavailable"
    assert contract_updates[-1]["payload"]["verification_authoritative"] is False


def test_interactive_mixed_workspace_neutral_config_path_keeps_repo_grounded_invalidation(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(
        repo,
        {
            "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
            "pyproject.toml": '[project]\nname = "demo"\nversion = "0.1.0"\n',
            ".env.example": "API_URL=\nUPLOAD_DIR=\n",
            "src/index.ts": "export const value = 1;\n",
            "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
        },
    )
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id="interactive-mixed-env-example-verify",
        cfg=AppConfig(model="test-model", verify_commands=["pytest -q"]),
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction="Update .env.example defaults.",
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == []
        assert session.verification_selection_source == "repo_scan.no_authoritative_commands"
        assert (
            session.verification_selection_reason
            == "repo scan found a mixed workspace without an authoritative verification surface"
        )
        assert session.verification_contract_type == "unavailable"
        assert session.verification_authoritative is False

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert result["commands"] == []
    assert result["summary"] == "verification skipped: no commands"
    assert result["all_passed"] is True

    events = list(read_session_events(sessions_dir / "interactive-mixed-env-example-verify.jsonl"))
    contract_updates = [
        event for event in events if event.get("type") == "verification_contract_updated"
    ]
    assert contract_updates
    assert contract_updates[-1]["payload"]["verification_selection_source"] == (
        "repo_scan.no_authoritative_commands"
    )
    assert (
        contract_updates[-1]["payload"]["verification_selection_reason"]
        == "repo scan found a mixed workspace without an authoritative verification surface"
    )
    assert contract_updates[-1]["payload"]["instruction_paths"] == [".env.example"]


@pytest.mark.parametrize(
    ("session_id", "instruction"),
    [
        ("interactive-js-package-json-verify", "Update package.json."),
        ("interactive-js-tsconfig-verify", "Update tsconfig.json."),
    ],
)
def test_interactive_js_bootstrap_paths_select_repo_native_build_command(
    tmp_path: Path,
    session_id: str,
    instruction: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(
        repo,
        {
            "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
            "tsconfig.json": '{\n  "compilerOptions": {\n    "strict": true\n  }\n}\n',
            "src/index.ts": "export const value = 1;\n",
        },
    )
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id=session_id,
        cfg=AppConfig(model="test-model", verify_commands=["pytest -q"]),
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction=instruction,
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == ["npm run build"]
        assert session.verification_selection_source == "repo_scan.likely_test_commands"
        assert session.verification_contract_type == "repo_native"
        assert session.verification_authoritative is True
    finally:
        session.close()

    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    contract_updates = [
        event for event in events if event.get("type") == "verification_contract_updated"
    ]
    assert contract_updates
    assert contract_updates[-1]["payload"]["verification_contract_type"] == "repo_native"
    assert contract_updates[-1]["payload"]["verification_authoritative"] is True


@pytest.mark.parametrize(
    ("instruction", "expected_reason"),
    [
        (
            "Refactor the parser.",
            "repo scan found a Python workspace without a discoverable test surface",
        ),
        (
            "Fix the bug.",
            "repo scan found a Python workspace without a discoverable test surface",
        ),
    ],
)
def test_interactive_python_repo_without_tests_marks_verification_unavailable(
    tmp_path: Path,
    instruction: str,
    expected_reason: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(
        repo,
        {
            "pyproject.toml": ('[project]\nname = "demo"\nversion = "0.1.0"\n'),
            "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
        },
    )
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id="interactive-python-no-tests",
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction=instruction,
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == []
        assert session.verification_selection_source == "repo_scan.no_authoritative_commands"
        assert session.verification_selection_reason == expected_reason
        assert session.verification_contract_type == "unavailable"
        assert session.verification_authoritative is False

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert result["commands"] == []
    assert result["summary"] == "verification skipped: no commands"
    assert result["all_passed"] is True


def test_verify_run_ignores_model_commands_when_verification_contract_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_commands: list[list[str]] = []

    def fake_run_task_verification(
        *,
        root: Path,
        commands: list[str],
        artifact_path: Path,
        cfg: AppConfig,
    ) -> VerifyRunResult:
        _ = root, cfg
        observed_commands.append(list(commands))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("verification skipped: no commands\n", encoding="utf-8")
        return VerifyRunResult(
            commands=list(commands), command_results=[], artifact_path=artifact_path
        )

    monkeypatch.setattr(agent_loop_mod, "run_task_verification", fake_run_task_verification)
    tools = _build_tools(
        tmp_path,
        cfg=AppConfig(model="test-model", verify_commands=[]),
        verify_command_selection=ResolvedVerifyCommands(
            commands=tuple(),
            source="repo_scan.no_authoritative_commands",
            reason="no configured verification command",
            contract_type="unavailable",
        ),
    )

    result = tools["verify_run"].run(
        {"commands": ['python -c "from calc import divide; divide(1, 0)"']}
    )

    assert observed_commands == [[]]
    assert result["commands"] == []
    assert result["summary"] == "verification skipped: no commands"
    assert result["all_passed"] is True
    assert result["ignored_model_verification_commands"] == [
        'python -c "from calc import divide; divide(1, 0)"'
    ]
    assert result["verification_skip_reason"] == "verification_contract_unavailable"


def test_interactive_plain_non_python_turn_does_not_require_generic_pytest(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "input.txt").write_text("data\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id="interactive-plain-no-generic-pytest",
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction="Create result.txt.",
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == []
        assert session.verification_selection_source == "repo_scan.no_authoritative_commands"
        assert session.verification_contract_type == "unavailable"
        assert session.verification_authoritative is False
        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert result["commands"] == []
    assert result["all_passed"] is True


def test_interactive_pathless_js_repo_with_repo_native_tests_keeps_authoritative_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(
        repo,
        {
            "package.json": '{"name":"demo-web","scripts":{"test":"vitest run"}}\n',
            "src/index.ts": "export const value = 1;\n",
        },
    )
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id="interactive-js-repo-native-verify",
    )
    try:
        assert session.effective_verification_commands == ["npm test"]
        assert session.verification_selection_source == "repo_scan.likely_test_commands"
        assert session.verification_contract_type == "repo_native"
        assert session.verification_authoritative is True

        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction="Fix the bug.",
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == ["npm test"]
        assert session.verification_selection_source == "repo_scan.likely_test_commands"
        assert session.verification_contract_type == "repo_native"
        assert session.verification_authoritative is True

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert [call for call in calls if call == "npm test"] == ["npm test"]
    assert result["commands"] == ["npm test"]
    assert result["all_passed"] is True


@pytest.mark.parametrize(
    ("session_id", "files", "expected_commands"),
    [
        (
            "interactive-mixed-js-repo-native-verify",
            {
                "package.json": '{"name":"demo-web","scripts":{"test":"vitest run"}}\n',
                "pyproject.toml": '[project]\nname = "demo"\nversion = "0.1.0"\n',
                "src/index.ts": "export const value = 1;\n",
                "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
            },
            ["npm test"],
        ),
        (
            "interactive-mixed-python-repo-native-verify",
            {
                "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
                "pyproject.toml": '[project]\nname = "demo"\nversion = "0.1.0"\n',
                "src/index.ts": "export const value = 1;\n",
                "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
                "tests/test_app.py": "def test_placeholder() -> None:\n    assert True\n",
            },
            ["pytest -q"],
        ),
    ],
)
def test_interactive_mixed_workspace_repo_native_commands_remain_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_id: str,
    files: dict[str, str],
    expected_commands: list[str],
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(repo, files)
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id=session_id,
        cfg=AppConfig(model="test-model", verify_commands=["pytest -q"]),
    )
    try:
        assert session.effective_verification_commands == expected_commands
        assert session.verification_selection_source == "repo_scan.likely_test_commands"
        assert session.verification_contract_type == "repo_native"
        assert session.verification_authoritative is True

        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction="Fix the bug.",
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == expected_commands
        assert session.verification_selection_source == "repo_scan.likely_test_commands"
        assert session.verification_contract_type == "repo_native"
        assert session.verification_authoritative is True

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert [call for call in calls if call in expected_commands] == expected_commands
    assert result["commands"] == expected_commands
    assert result["all_passed"] is True


@pytest.mark.parametrize(
    (
        "session_id",
        "instruction",
        "expected_path",
        "expected_commands",
        "expected_source",
        "expected_contract_type",
        "expected_authoritative",
        "expected_reason",
    ),
    [
        (
            "interactive-mixed-js-target-verify",
            "Update src/index.ts auth handling.",
            "src/index.ts",
            ["npm run build"],
            "repo_scan.likely_test_commands",
            "repo_native",
            True,
            "repo scan discovered package.json verification scripts",
        ),
        (
            "interactive-mixed-python-target-verify",
            "Improve upload logging in src/demo/app.py",
            "src/demo/app.py",
            [],
            "task_refinement.no_authoritative_commands",
            "unavailable",
            False,
            "Python task has no discoverable test surface, so generic pytest is not trusted",
        ),
    ],
)
def test_interactive_mixed_workspace_explicit_targets_keep_task_specific_selection_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_id: str,
    instruction: str,
    expected_path: str,
    expected_commands: list[str],
    expected_source: str,
    expected_contract_type: str,
    expected_authoritative: bool,
    expected_reason: str,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(
        repo,
        {
            "package.json": '{"name":"demo-web","scripts":{"build":"vite build"}}\n',
            "pyproject.toml": '[project]\nname = "demo"\nversion = "0.1.0"\n',
            "src/index.ts": "export const value = 1;\n",
            "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
        },
    )
    sessions_dir = tmp_path / "sessions"

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id=session_id,
        cfg=AppConfig(model="test-model", verify_commands=["pytest -q"]),
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction=instruction,
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == expected_commands
        assert session.verification_selection_source == expected_source
        assert session.verification_selection_reason == expected_reason
        assert session.verification_contract_type == expected_contract_type
        assert session.verification_authoritative is expected_authoritative

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert [call for call in calls if call in expected_commands] == expected_commands
    assert result["commands"] == expected_commands
    assert result["all_passed"] is True

    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    contract_updates = [
        event for event in events if event.get("type") == "verification_contract_updated"
    ]
    assert contract_updates
    assert contract_updates[-1]["payload"]["verification_selection_source"] == expected_source
    assert contract_updates[-1]["payload"]["verification_selection_reason"] == expected_reason
    assert contract_updates[-1]["payload"]["instruction_paths"] == [expected_path]


def test_interactive_repo_native_verification_selection_stays_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(
        repo,
        {
            "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
            "tests/test_app.py": "def test_placeholder() -> None:\n    assert True\n",
        },
    )
    sessions_dir = tmp_path / "sessions"
    cfg = AppConfig(model="test-model", verify_commands=["pytest tests/test_app.py -q"])

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id="interactive-repo-native-verify",
        cfg=cfg,
    )
    try:
        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction="Fix the bug.",
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == ["pytest tests/test_app.py -q"]
        assert session.verification_selection_source == "config.verify_commands"
        assert session.verification_contract_type == "repo_native"
        assert session.verification_authoritative is True

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert [call for call in calls if call == "pytest tests/test_app.py -q"] == [
        "pytest tests/test_app.py -q"
    ]
    assert result["commands"] == ["pytest tests/test_app.py -q"]
    assert result["all_passed"] is True


def test_interactive_explicit_verify_override_wins_and_survives_task_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_files(
        repo,
        {
            "README.md": "# Demo\n",
            "package.json": '{"name":"demo-web","scripts":{"test":"vitest run"}}\n',
            "pyproject.toml": '[project]\nname = "demo"\nversion = "0.1.0"\n',
            "src/demo/app.py": "def handler() -> str:\n    return 'ok'\n",
        },
    )
    sessions_dir = tmp_path / "sessions"
    verify_cmd = ["pnpm --dir packages/web test -- --runInBand"]

    session = _create_interactive_session(
        repo,
        sessions_dir=sessions_dir,
        session_id="interactive-explicit-verify",
        verify_cmd=verify_cmd,
    )
    try:
        assert session.effective_verification_commands == verify_cmd
        assert session.verification_selection_source == "cli.verify_cmd"
        assert session.verification_contract_type == "explicit_override"
        assert session.verification_authoritative is True

        agent_loop_mod._refresh_interactive_turn_verification_selection(
            session,
            instruction="Fix the bug.",
            route_execution_posture="execute",
        )

        assert session.effective_verification_commands == verify_cmd
        assert session.verification_selection_source == "cli.verify_cmd"
        assert session.verification_contract_type == "explicit_override"
        assert session.verification_authoritative is True

        result = session.tools["verify_run"].run({})
    finally:
        session.close()

    assert [call for call in calls if call in verify_cmd] == verify_cmd
    assert result["commands"] == verify_cmd
    assert result["all_passed"] is True

    events = list(read_session_events(sessions_dir / "interactive-explicit-verify.jsonl"))
    contract_updates = [
        event for event in events if event.get("type") == "verification_contract_updated"
    ]
    assert contract_updates == []


def test_verify_run_allows_override_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q"]
    tools = _build_tools(tmp_path, cfg=cfg)

    result = tools["verify_run"].run({"commands": ["python -m pytest -q", "ruff check src"]})

    assert calls == ["python -m pytest -q", "ruff check src"]
    assert result["commands"] == ["python -m pytest -q", "ruff check src"]
    assert result["all_passed"] is True
    assert result["failed_commands"] == []
    assert result["summary"] == "verification passed (2/2)"


@pytest.mark.parametrize(
    "requested_command",
    [
        "python -m pytest -q && ruff check src",
        "python -m pytest -q; ruff check src",
    ],
)
def test_verify_run_splits_simple_chained_override_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_command: str,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    tools = _build_tools(tmp_path)

    result = tools["verify_run"].run({"commands": [requested_command]})

    assert calls == ["python -m pytest -q", "ruff check src"]
    assert result["commands"] == ["python -m pytest -q", "ruff check src"]
    assert result["all_passed"] is True


def test_verify_run_splits_chained_override_before_contract_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    tools = _build_tools(
        tmp_path,
        effective_verification_commands=["pytest -q", "ruff check src"],
    )

    result = tools["verify_run"].run({"commands": ["python -m pytest -q && ruff check src"]})

    assert calls == ["python -m pytest -q", "ruff check src"]
    assert result["commands"] == ["python -m pytest -q", "ruff check src"]
    assert result["all_passed"] is True


def test_verify_run_rejects_non_verifier_inside_simple_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for unsafe commands"),
    )

    tools = _build_tools(tmp_path)

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="disallowed_shell_control_flow",
    ):
        tools["verify_run"].run({"commands": ["pytest -q && echo ok"]})


def test_verify_run_rejects_incompatible_override_against_effective_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for incompatible overrides"),
    )

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["pytest -q"],
    )

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="verify_run commands must stay within the session's effective verification contract.",
    ):
        tools["verify_run"].run({"commands": ['python3 -c "print(123)"']})


def test_verify_run_allows_targeted_override_against_effective_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["pytest -q"],
    )

    result = tools["verify_run"].run({"commands": ["pytest tests/test_cli.py -q"]})

    assert calls == ["pytest tests/test_cli.py -q"]
    assert result["commands"] == ["pytest tests/test_cli.py -q"]
    assert result["all_passed"] is True


@pytest.mark.parametrize(
    "requested_command",
    [
        "pytest tests/test_cli.py -v",
        "python -m pytest tests/test_cli.py -v",
        "cd /tmp/x && python -m pytest tests/test_batching.py -v",
        "bash -lc 'python -m pytest tests/test_cli.py -v'",
        "poetry run python -m pytest tests/test_cli.py -v",
        "uv run pytest tests/test_cli.py -v",
        "pipenv run python -m pytest tests/test_cli.py -v",
    ],
)
def test_verify_run_allows_targeted_pytest_variants_against_effective_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_command: str,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["pytest -q"],
    )

    result = tools["verify_run"].run({"commands": [requested_command]})

    assert calls == [requested_command]
    assert result["commands"] == [requested_command]
    assert result["all_passed"] is True


def test_verify_run_marks_go_no_tests_to_run_as_skipped_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\texample/pkg\t0.002s [no tests to run]\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["go test ./..."]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["go test ./..."],
    )

    result = tools["verify_run"].run({"commands": ["go test -run NonExistent ./..."]})

    assert calls == ["go test -run NonExistent ./..."]
    assert result["all_passed"] is True
    assert result["failed_commands"] == []
    assert result["summary"] == "verification skipped: nothing to verify (1/1)"
    assert result["command_results"][0]["status"] == "skipped"
    assert result["command_results"][0]["ok"] is True
    assert result["command_results"][0]["real_execution"] is False
    assert result["command_results"][0]["non_execution_reason"] == "go_test_no_tests_to_run"


def test_verify_run_marks_go_no_test_files_as_skipped_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="?   \texample/pkg\t[no test files]\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["go test ./..."]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["go test ./..."],
    )

    result = tools["verify_run"].run({"commands": ["go test ./..."]})

    assert calls == ["go test ./..."]
    assert result["all_passed"] is True
    assert result["failed_commands"] == []
    assert result["summary"] == "verification skipped: nothing to verify (1/1)"
    assert result["command_results"][0]["status"] == "skipped"
    assert result["command_results"][0]["ok"] is True
    assert result["command_results"][0]["real_execution"] is False
    assert result["command_results"][0]["non_execution_reason"] == "go_test_no_test_files"


def test_verify_run_accepts_mixed_go_package_output_with_real_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(
            returncode=0,
            stdout=("?   \texample/pkg1\t[no test files]\nok  \texample/pkg2\t0.002s\n"),
        )

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["go test ./..."]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["go test ./..."],
    )

    result = tools["verify_run"].run({"commands": ["go test ./..."]})

    assert calls == ["go test ./..."]
    assert result["all_passed"] is True
    assert result["failed_commands"] == []
    assert result["summary"] == "verification passed (1/1)"
    assert result["command_results"][0]["real_execution"] is True
    assert result["command_results"][0].get("non_execution_reason") is None


def test_verify_run_marks_unittest_zero_tests_as_skipped_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(
            returncode=0,
            stderr="----------------------------------------------------------------------\nRan 0 tests in 0.000s\n\nOK\n",
        )

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["python -m unittest discover -s tests"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["python -m unittest discover -s tests"],
    )

    result = tools["verify_run"].run({})

    assert calls == ["python -m unittest discover -s tests"]
    assert result["all_passed"] is True
    assert result["failed_commands"] == []
    assert result["summary"] == "verification skipped: nothing to verify (1/1)"
    assert result["command_results"][0]["status"] == "skipped"
    assert result["command_results"][0]["ok"] is True
    assert result["command_results"][0]["real_execution"] is False
    assert result["command_results"][0]["non_execution_reason"] == "unittest_no_tests_run"


def test_verify_run_marks_maven_zero_tests_as_skipped_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(
            returncode=0,
            stdout=(
                "[INFO] Running example.EmptyTest\n"
                "Tests run: 0, Failures: 0, Errors: 0, Skipped: 0\n"
                "[INFO] BUILD SUCCESS\n"
            ),
        )

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["mvn test"]
    tools = _build_tools(tmp_path, cfg=cfg, effective_verification_commands=["mvn test"])

    result = tools["verify_run"].run({})

    assert calls == ["mvn test"]
    assert result["all_passed"] is True
    assert result["failed_commands"] == []
    assert result["summary"] == "verification skipped: nothing to verify (1/1)"
    assert result["command_results"][0]["status"] == "skipped"
    assert result["command_results"][0]["ok"] is True
    assert result["command_results"][0]["real_execution"] is False
    assert result["command_results"][0]["non_execution_reason"] == "maven_test_zero_tests"


def test_verify_run_classifies_missing_wrapper_as_infra_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        assert str(cmd) == "./gradlew test"
        return _cp(returncode=127, stderr="/bin/bash: ./gradlew: No such file or directory\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["./gradlew test"]
    tools = _build_tools(tmp_path, cfg=cfg, effective_verification_commands=["./gradlew test"])

    result = tools["verify_run"].run({})

    assert result["all_passed"] is False
    assert result["failure_category"] == "infra_unavailable"
    assert result["command_results"][0]["real_execution"] is False
    assert result["command_results"][0]["non_execution_reason"] == "execution_layer_failure"
    assert result["primary_failure"]["snippet"] == (
        "/bin/bash: ./gradlew: No such file or directory"
    )


def test_verify_run_allows_exact_unknown_family_command_within_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["ruby -Ilib:test test/**/*_test.rb"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["ruby -Ilib:test test/**/*_test.rb"],
    )

    result = tools["verify_run"].run({"commands": ["ruby -Ilib:test test/**/*_test.rb"]})

    assert calls == ["ruby -Ilib:test test/**/*_test.rb"]
    assert result["all_passed"] is True


def test_verify_run_expands_recursive_globs_portably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_repo_files(tmp_path, {"test/service_test.rb": "puts 'ok'\n"})
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="1 runs, 1 assertions, 0 failures\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["ruby -Ilib:test test/**/*_test.rb"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["ruby -Ilib:test test/**/*_test.rb"],
    )

    result = tools["verify_run"].run({})

    assert calls == ["ruby -Ilib:test test/service_test.rb"]
    assert result["all_passed"] is True
    assert result["command_results"][0]["command"] == "ruby -Ilib:test test/**/*_test.rb"
    assert result["command_results"][0]["effective_command"] == (
        "ruby -Ilib:test test/service_test.rb"
    )


def test_verify_run_rejects_different_family_override_against_effective_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for incompatible overrides"),
    )

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["pytest -q"],
    )

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="verify_run commands must stay within the session's effective verification contract.",
    ):
        tools["verify_run"].run({"commands": ["ruff check src"]})


def test_verify_run_rejects_npm_lint_override_against_effective_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for incompatible overrides"),
    )

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["npm test"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["npm test"],
    )

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="verify_run commands must stay within the session's effective verification contract.",
    ):
        tools["verify_run"].run({"commands": ["npm run lint"]})


@pytest.mark.parametrize(
    ("effective_commands", "requested_command"),
    [
        (["cargo test"], "cargo test --no-run"),
        (["cargo test"], "cargo test -- --list"),
        (["cargo test"], "cargo test -- --list --format terse"),
        (["pytest -q"], "pytest -q --setup-plan"),
        (["pytest -q"], "pytest -q --co"),
        (["pytest -q"], "py.test --co -q"),
        (["pytest -q"], "pytest --help"),
        (["pytest -q"], "pytest -q --help"),
        (["go test ./..."], "go test -c ./..."),
        (["go test ./..."], "go test -list . ./..."),
        (["go test ./..."], "go test -run '^$' ./..."),
        (["ruff check ."], "ruff check --fix src/app.py"),
        (["mypy"], "mypy --install-types"),
    ],
)
def test_verify_run_rejects_non_executing_override_against_effective_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    effective_commands: list[str],
    requested_command: str,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for non-executing overrides"),
    )

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = list(effective_commands)
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=effective_commands,
    )

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="non_assertive_verification_mode",
    ):
        tools["verify_run"].run({"commands": [requested_command]})


def test_verify_run_rejects_divergent_override_in_managed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(_cmd, **_kwargs):  # type: ignore[no-untyped-def]
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        authoritative_verification_commands=["pytest -q"],
    )

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="Managed verification commands are locked to the authoritative Forge command set.",
    ):
        tools["verify_run"].run({"commands": ["ruff check ."]})


def test_verify_run_allows_identical_override_in_managed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(cmd))
        return _cp(returncode=0, stdout="ok\n")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model")
    cfg.verify_commands = ["pytest -q"]
    tools = _build_tools(
        tmp_path,
        cfg=cfg,
        authoritative_verification_commands=["pytest -q"],
    )

    result = tools["verify_run"].run({"commands": ["pytest -q"]})

    assert calls == ["pytest -q"]
    assert result["commands"] == ["pytest -q"]
    assert result["all_passed"] is True


def test_verify_run_rejects_compound_shell_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for unsafe commands"),
    )

    tools = _build_tools(tmp_path)

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="disallowed_shell_control_flow",
    ):
        tools["verify_run"].run({"commands": ["pytest -q || true"]})


def test_verify_run_rejects_explicit_trusted_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for unsafe pipeline"),
    )
    pipeline = "tool args | tail -n 1"
    tools = _build_tools(
        tmp_path,
        verify_command_selection=ResolvedVerifyCommands(
            commands=(pipeline,),
            source="task_refinement.explicit_user_command",
            contract_type="task_acceptance",
        ),
    )

    with pytest.raises(verify_gate_mod.VerifyError, match="unsafe_pipeline"):
        tools["verify_run"].run({})


def test_verify_run_rejects_arbitrary_model_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for untrusted pipeline"),
    )
    tools = _build_tools(tmp_path)

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="unsafe_pipeline",
    ):
        tools["verify_run"].run({"commands": ["printf ok | cat"]})


def test_verify_run_authoritative_pipeline_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for unsafe pipeline"),
    )
    pipeline = "tool args | tail -n 1"
    tools = _build_tools(
        tmp_path,
        authoritative_verification_commands=[pipeline],
    )

    with pytest.raises(verify_gate_mod.VerifyError, match="unsafe_pipeline"):
        tools["verify_run"].run({"commands": [pipeline]})


def test_verify_run_malformed_authoritative_command_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run malformed host command"),
    )
    tools = _build_tools(
        tmp_path,
        authoritative_verification_commands=["pytest -q '"],
    )

    with pytest.raises(
        verify_gate_mod.VerifyError,
        match="authoritative verification command is invalid",
    ):
        tools["verify_run"].run({})


def test_verify_run_truncates_output_preview_but_keeps_full_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big_output = "x" * 900

    def fake_run(_cmd, **_kwargs):  # type: ignore[no-untyped-def]
        return _cp(returncode=0, stdout=big_output)

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    tools = _build_tools(tmp_path)
    result = tools["verify_run"].run({"commands": ["pytest -q"]})

    command_result = result["command_results"][0]
    assert command_result["output_chars"] == 900
    assert command_result["output_truncated"] is True
    assert len(command_result["output_preview"]) <= verify_gate_mod.VERIFY_OUTPUT_PREVIEW_CHARS
    assert command_result["output_preview"].endswith("...(truncated)")

    artifact = tmp_path / str(result["artifact_path"])
    assert result["artifact_saved"] is True
    assert result["artifact_readable_via_fs"] is True
    assert result["artifact_location"] == "workspace_root"
    assert artifact.read_text(encoding="utf-8").count("x") >= 900


def test_verify_run_blocks_in_readonly_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run in readonly mode"),
    )
    tools = _build_tools(tmp_path, mode="readonly")

    assert "verify_run" not in tools


def test_verify_run_requires_review_mode_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run when approval is denied"),
    )
    tools = _build_tools(
        tmp_path,
        mode="review",
        surface=NoopSurface(),
        yes=False,
        non_interactive=False,
    )

    with pytest.raises(AgentRuntimeError, match="User declined: verify_run"):
        tools["verify_run"].run({"commands": ["pytest -q"]})
