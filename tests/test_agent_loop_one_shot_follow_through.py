from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

import sylliptor_agent_cli.agent_loop as agent_loop_mod
from sylliptor_agent_cli.agent_loop import (
    ToolDef,
    TurnExecutionState,
    _assistant_text_contains_progress_intent,
    _assistant_text_has_blocker_marker,
    _assistant_text_has_completion_marker,
    _classify_one_shot_repo_turn_intent,
    _completion_gate_problems,
    _matching_effective_verification_commands,
    _normalize_marker_text,
    _record_tool_effect,
    _rewrite_final_summary_for_language,
    _runtime_message,
    _shell_command_is_verification_attempt,
    _verification_expected_for_turn,
    create_session,
)
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.session_store import read_session_events
from sylliptor_agent_cli.surface.noop_surface import NoopSurface
from sylliptor_agent_cli.verify_gate import VerifyCommandResult, VerifyRunResult


class _ScriptedClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls = 0
        self.call_records: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = on_text_delta, temperature
        self.call_records.append(
            {
                "messages": list(messages),
                "tools": tools,
                "stream": stream,
            }
        )
        response = self._responses[self.calls]
        self.calls += 1
        return response


class _RecordingSurface(NoopSurface):
    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []
        self.progress_updates: list[str] = []
        self.final_messages: list[str] = []

    def on_error(self, err: str) -> None:
        self.errors.append(err)

    def on_progress_update(self, message: str) -> None:
        self.progress_updates.append(message)

    def on_assistant_message_done(self, text: str) -> None:
        self.final_messages.append(text)


_VERIFY_OK_COMMAND = "pytest tests/test_cli.py -q"
_VERIFY_FAIL_COMMAND = "pytest tests/failing_case.py -q"
_VERIFY_IMPORT_ERROR_COMMAND = "pytest tests/search_notes_import_error.py -q"
_VERIFY_NAME_ERROR_COMMAND = "pytest tests/search_notes_name_error.py -q"
_VERIFY_RUFF_OK_COMMAND = "ruff check src/app.py"
_VERIFY_GO_OK_COMMAND = "go test ./pkg/..."
_VERIFY_GO_NO_TESTS_COMMAND = "go test -run NonExistent ./..."
_VERIFY_GO_NO_TEST_FILES_COMMAND = "go test ./..."
_VERIFY_GO_MIXED_OK_COMMAND = "go test ./mixed/..."


def _fake_verify_command_result(command: str) -> VerifyCommandResult:
    normalized = " ".join(str(command).split())
    if normalized == _VERIFY_OK_COMMAND:
        return VerifyCommandResult(command=command, exit_code=0, output="ok\n")
    if normalized == _VERIFY_FAIL_COMMAND:
        return VerifyCommandResult(
            command=command,
            exit_code=1,
            output="FAILED tests/failing_case.py::test_failure - AssertionError\n",
        )
    if normalized == _VERIFY_IMPORT_ERROR_COMMAND:
        return VerifyCommandResult(
            command=command,
            exit_code=1,
            output="ImportError: cannot import name search_notes\n",
        )
    if normalized == _VERIFY_NAME_ERROR_COMMAND:
        return VerifyCommandResult(
            command=command,
            exit_code=1,
            output="NameError: search_notes is not defined\n",
        )
    if normalized == _VERIFY_RUFF_OK_COMMAND:
        return VerifyCommandResult(
            command=command,
            exit_code=0,
            output="All checks passed!\n",
            real_execution=True,
        )
    if normalized == _VERIFY_GO_OK_COMMAND:
        return VerifyCommandResult(
            command=command,
            exit_code=0,
            output="ok\texample/pkg\t0.002s\n",
            real_execution=True,
        )
    if normalized == _VERIFY_GO_NO_TESTS_COMMAND:
        return VerifyCommandResult(
            command=command,
            exit_code=0,
            output="ok\texample/pkg\t0.002s [no tests to run]\n",
            real_execution=False,
            non_execution_reason="go_test_no_tests_to_run",
        )
    if normalized == _VERIFY_GO_NO_TEST_FILES_COMMAND:
        return VerifyCommandResult(
            command=command,
            exit_code=0,
            output="?   \texample/pkg\t[no test files]\n",
            real_execution=False,
            non_execution_reason="go_test_no_test_files",
        )
    if normalized == _VERIFY_GO_MIXED_OK_COMMAND:
        return VerifyCommandResult(
            command=command,
            exit_code=0,
            output="?   \texample/pkg1\t[no test files]\nok  \texample/pkg2\t0.002s\n",
            real_execution=True,
        )
    return VerifyCommandResult(
        command=command,
        exit_code=1,
        output=f"unexpected verify command in test: {command}\n",
    )


def test_verification_failure_snippet_prefers_structured_primary_failure() -> None:
    snippet = agent_loop_mod.extract_verification_failure_snippet(
        tool_name="verify_run",
        result={
            "summary": "verification failed (0/1); failed: pytest tests/search_notes_name_error.py -q",
            "failed_commands": ["pytest tests/search_notes_name_error.py -q"],
            "primary_failure": {
                "command": "pytest tests/search_notes_name_error.py -q",
                "effective_command": "pytest tests/search_notes_name_error.py -q",
                "snippet": "NameError: search_notes is not defined",
                "output_truncated": False,
                "fallback_used": False,
            },
            "command_results": [
                {
                    "command": "pytest tests/search_notes_name_error.py -q",
                    "effective_command": "pytest tests/search_notes_name_error.py -q",
                    "exit_code": 1,
                    "ok": False,
                    "output_preview": "verification failed\n",
                }
            ],
        },
    )

    assert snippet == "NameError: search_notes is not defined"


def _latest_completion_gate_payload(
    sessions_dir: Path,
    session_id: str,
) -> dict[str, Any]:
    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    return dict(incomplete_events[-1].get("payload") or {})


def _assert_forced_final_summary_emitted(events: list[dict[str, Any]]) -> str:
    assert any(event.get("type") == "forced_final_summary_requested" for event in events)
    assert any(
        event.get("type") in {"forced_final_summary_completed", "forced_final_summary_fallback"}
        for event in events
    )
    assistant_events = [event for event in events if event.get("type") == "assistant_message"]
    final_events = [event for event in events if event.get("type") == "final"]
    assert assistant_events
    assert final_events
    content = str((assistant_events[-1].get("payload") or {}).get("content") or "")
    assert content
    assert str((final_events[-1].get("payload") or {}).get("content") or "") == content
    return content


def _assert_last_forced_summary_request(
    client: _ScriptedClient,
    *,
    latest_assistant_text: str,
    termination_cause: str,
) -> None:
    assert client.call_records
    request = client.call_records[-1]
    assert request["tools"] is None
    request_messages = request["messages"]
    assert request_messages[-2] == {
        "role": "assistant",
        "content": latest_assistant_text,
    }
    assert str(request_messages[-1].get("role")) == "system"
    assert f"Stop reason: {termination_cause}" in str(request_messages[-1].get("content"))


def test_completion_gate_remembers_failed_configured_command_after_subset_pass(
    tmp_path: Path,
) -> None:
    commands = ["go test ./...", "go run ./cmd/importer fixtures/mixed.csv"]
    state = TurnExecutionState(
        execution_requested=True,
        expected_verification_commands=set(commands),
        material_edit_count=1,
        touched_repo_paths={"cmd/importer/main.go"},
    )

    _record_tool_effect(
        root=tmp_path,
        state=state,
        tool_name="verify_run",
        arguments={"commands": commands},
        status="ok",
        result={
            "commands": commands,
            "command_results": [
                {
                    "command": "go test ./...",
                    "effective_command": "go test ./...",
                    "ok": True,
                    "exit_code": 0,
                    "real_execution": True,
                    "output_preview": "ok\texample/importer\t0.003s\n",
                },
                {
                    "command": "go run ./cmd/importer fixtures/mixed.csv",
                    "effective_command": "go run ./cmd/importer fixtures/mixed.csv",
                    "ok": False,
                    "exit_code": 1,
                    "real_execution": None,
                    "output_preview": "exit status 1\n",
                },
            ],
        },
        known_verification_commands=commands,
    )
    assert state.failed_verification_commands() == {"go run ./cmd/importer fixtures/mixed.csv"}

    _record_tool_effect(
        root=tmp_path,
        state=state,
        tool_name="verify_run",
        arguments={"commands": ["go test ./..."]},
        status="ok",
        result={
            "commands": ["go test ./..."],
            "command_results": [
                {
                    "command": "go test ./...",
                    "effective_command": "go test ./...",
                    "ok": True,
                    "exit_code": 0,
                    "real_execution": True,
                    "output_preview": "ok\texample/importer\t(cached)\n",
                }
            ],
            "all_passed": True,
        },
        known_verification_commands=commands,
    )

    assert state.covered_verification_commands == {"go test ./..."}
    assert state.failed_verification_commands() == {"go run ./cmd/importer fixtures/mixed.csv"}
    assert "go run ./cmd/importer fixtures/mixed.csv" in state.first_failed_verification_snippet()
    problems = _completion_gate_problems(
        state=state,
        final_text="Implemented and verified.",
        blocked=False,
        verification_expected=True,
        require_material_edit_evidence=False,
    )
    assert "verification_failed" in problems
    assert "verification_incomplete" not in problems


def test_verify_run_coverage_uses_original_command_when_effective_command_expands_glob(
    tmp_path: Path,
) -> None:
    configured = "ruby -Ilib:test test/**/*_test.rb"
    state = TurnExecutionState(
        execution_requested=True,
        expected_verification_commands={configured},
        material_edit_count=1,
        touched_repo_paths={"lib/service.rb"},
    )

    _record_tool_effect(
        root=tmp_path,
        state=state,
        tool_name="verify_run",
        arguments={},
        status="ok",
        result={
            "commands": [configured],
            "command_results": [
                {
                    "command": configured,
                    "effective_command": "ruby -Ilib:test test/service_test.rb",
                    "ok": True,
                    "exit_code": 0,
                    "real_execution": True,
                    "output_preview": "1 runs, 1 assertions, 0 failures\n",
                }
            ],
            "all_passed": True,
        },
        known_verification_commands=[configured],
    )

    assert state.covered_verification_commands == {configured}
    assert state.failed_verification_commands() == set()
    problems = _completion_gate_problems(
        state=state,
        final_text="Implemented and verified.",
        blocked=False,
        verification_expected=True,
        require_material_edit_evidence=False,
    )
    assert problems == []


def _write_repo_text(root: Path, rel_path: str, content: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _init_git_repo_with_commit(repo: Path) -> None:
    repo.mkdir()
    subprocess.run(
        ["git", "-C", os.fspath(repo), "init"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", os.fspath(repo), "config", "user.name", "Test User"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", os.fspath(repo), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "README.md").write_text("repo\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", os.fspath(repo), "add", "README.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", os.fspath(repo), "commit", "-m", "init"],
        check=True,
        capture_output=True,
        text=True,
    )


def _shell_run_with_repo_mutations(
    *,
    mutations: dict[str, tuple[str, str]],
    exit_codes: dict[str, int] | None = None,
    stdouts: dict[str, str] | None = None,
    stderrs: dict[str, str] | None = None,
):
    command_exit_codes = exit_codes or {}
    command_stdouts = stdouts or {}
    command_stderrs = stderrs or {}

    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = cwd, runner
        if cmd in mutations:
            rel_path, content = mutations[cmd]
            _write_repo_text(root, rel_path, content)
            exit_code = command_exit_codes.get(cmd, 0)
            return {
                "cmd": cmd,
                "effective_cmd": cmd,
                "exit_code": exit_code,
                "stdout": command_stdouts.get(cmd, "mutated\n" if exit_code == 0 else ""),
                "stderr": command_stderrs.get(
                    cmd,
                    "" if exit_code == 0 else "command failed\n",
                ),
            }
        if cmd == "pytest -q":
            _write_repo_text(root, ".pytest_cache/v/cache/nodeids", "cache\n")
            return {
                "cmd": cmd,
                "effective_cmd": cmd,
                "exit_code": 0,
                "stdout": "ok\n",
                "stderr": "",
            }
        if cmd == "ruff check .":
            _write_repo_text(root, ".ruff_cache/CACHEDIR.TAG", "cache\n")
            return {
                "cmd": cmd,
                "effective_cmd": cmd,
                "exit_code": 0,
                "stdout": "All checks passed!\n",
                "stderr": "",
            }
        return {"cmd": cmd, "effective_cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    return fake_shell_run


def _verify_run_with_repo_mutations(
    *,
    mutations_by_call: dict[int, list[tuple[str, str]]] | None = None,
    mutations_by_command: dict[str, list[tuple[str, str]]] | None = None,
):
    call_count = 0
    per_call_mutations = mutations_by_call or {}
    per_command_mutations = {
        " ".join(command.split()): list(mutations)
        for command, mutations in (mutations_by_command or {}).items()
    }

    def fake_run_task_verification(
        *,
        root: Path,
        commands: list[str],
        artifact_path: Path,
        cfg: AppConfig,
    ) -> VerifyRunResult:
        nonlocal call_count
        _ = cfg
        call_count += 1
        for rel_path, content in per_call_mutations.get(call_count, []):
            _write_repo_text(root, rel_path, content)
        for command in commands:
            normalized = " ".join(str(command).split())
            for rel_path, content in per_command_mutations.get(normalized, []):
                _write_repo_text(root, rel_path, content)
        command_results = [_fake_verify_command_result(command) for command in commands]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_body = "\n\n".join(
            [f"$ {item.command}\n{item.output}".rstrip() for item in command_results]
        )
        artifact_path.write_text(artifact_body + "\n", encoding="utf-8")
        return VerifyRunResult(
            commands=list(commands),
            command_results=command_results,
            artifact_path=artifact_path,
        )

    return fake_run_task_verification


@pytest.fixture(autouse=True)
def _fake_one_shot_verify_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_task_verification(
        *,
        root: Path,
        commands: list[str],
        artifact_path: Path,
        cfg: AppConfig,
    ) -> VerifyRunResult:
        _ = root, cfg
        command_results = [_fake_verify_command_result(command) for command in commands]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_body = "\n\n".join(
            [f"$ {item.command}\n{item.output}".rstrip() for item in command_results]
        )
        artifact_path.write_text(artifact_body + "\n", encoding="utf-8")
        return VerifyRunResult(
            commands=list(commands),
            command_results=command_results,
            artifact_path=artifact_path,
        )

    monkeypatch.setattr(agent_loop_mod, "run_task_verification", fake_run_task_verification)


@pytest.mark.parametrize(
    "cmd",
    [
        "echo pytest -q",
        "printf 'make verify\\n'",
        "python -c \"print('pytest -q')\"",
        "true # pytest -q",
        "bash -lc 'echo pytest -q'",
        "pytest -q || true",
        "pytest -q ; true",
        "pytest -q\ntrue",
        "bash -lc 'pytest -q || true'",
        "make verify || true",
        "npm test -- redirect || true",
        "cargo test redirect --quiet ; true",
        "cargo test --no-run",
        "cargo test -- --list",
        "cargo test -- --list --format terse",
        "go test -c ./...",
        "go test -list . ./...",
        "go test -run '^$' ./...",
        "pytest -q --setup-plan",
        "pytest -q --co",
        "py.test --co -q",
        "pytest --help",
        "pytest -q --help",
        "ruff check --fix src/app.py",
        "mypy --install-types",
        "pytest --version",
        "npm run lint",
        "cargo bench",
        "make lint",
    ],
)
def test_shell_verification_detection_rejects_non_matching_or_non_verification_commands(
    cmd: str,
) -> None:
    known = ["pytest -q", "make verify"]
    if cmd == "npm run lint":
        known = ["npm test"]
    elif cmd == "npm test -- redirect || true":
        known = ["npm test"]
    elif cmd == "cargo bench":
        known = ["cargo test"]
    elif cmd == "cargo test redirect --quiet ; true":
        known = ["cargo test"]
    elif cmd == "cargo test --no-run":
        known = ["cargo test"]
    elif cmd in {"cargo test -- --list", "cargo test -- --list --format terse"}:
        known = ["cargo test"]
    elif cmd in {"go test -c ./...", "go test -list . ./..."}:
        known = ["go test ./..."]
    elif cmd == "go test -run '^$' ./...":
        known = ["go test ./..."]
    elif cmd == "pytest -q --setup-plan":
        known = ["pytest -q"]
    elif cmd in {"pytest -q --co", "py.test --co -q"}:
        known = ["pytest -q"]
    elif cmd == "ruff check --fix src/app.py":
        known = ["ruff check ."]
    elif cmd == "mypy --install-types":
        known = ["mypy"]
    assert (
        _shell_command_is_verification_attempt(
            cmd,
            known_verification_commands=known,
        )
        is False
    )


@pytest.mark.parametrize(
    ("cmd", "known"),
    [
        ("pytest -q", ["pytest -q"]),
        ("make verify", ["make verify"]),
        ("just verify", ["just verify"]),
        ("PYTHONPATH=src pytest -q", ["pytest -q"]),
        ("env PYTHONPATH=src pytest -q", ["pytest -q"]),
        ("python -m pytest -q", ["pytest -q"]),
        ("python3 -m pytest -q", ["pytest -q"]),
        ("py -m pytest -q", ["pytest -q"]),
        ("uv run py -m pytest -q", ["pytest -q"]),
        ("pytest tests/test_cli.py -q", ["pytest -q"]),
        ("pytest tests/test_cli.py -v", ["pytest -q"]),
        ("python -m pytest tests/test_cli.py -v", ["pytest -q"]),
        ("python -m pytest tests/test_cli.py -q", ["pytest -q"]),
        ("python3 -m pytest tests/test_cli.py -q", ["pytest -q"]),
        ("py -m pytest tests/test_cli.py -q", ["pytest -q"]),
        ("cd /tmp/x && python -m pytest tests/test_batching.py -v", ["pytest -q"]),
        ("python3 -m unittest -v", ["unittest -v"]),
        ("bash -lc 'pytest -q'", ["pytest -q"]),
        ("bash -lc 'cd /tmp/x && python -m pytest tests/test_batching.py -v'", ["pytest -q"]),
        ("bash -lc 'PYTHONPATH=src pytest -q'", ["pytest -q"]),
        ("bash -lc 'make verify'", ["make verify"]),
        ("poetry run pytest -q", ["pytest -q"]),
        ("uv run pytest -q", ["pytest -q"]),
        ("pipenv run pytest -q", ["pytest -q"]),
        ("python -m ruff check .", ["ruff check ."]),
        ("python3 -m ruff check .", ["ruff check ."]),
        ("py -m ruff check .", ["ruff check ."]),
        ("uv run python -m ruff check .", ["ruff check ."]),
        ("uv run py -m ruff check .", ["ruff check ."]),
        (r"C:\Python311\python.exe -m pytest -q", ["pytest -q"]),
        (r"C:\Python311\python.exe -m ruff check .", ["ruff check ."]),
        (r'"C:\Program Files\Python311\python.exe" -m pytest -q', ["pytest -q"]),
        (r'"C:\Program Files\Python311\python.exe" -m ruff check .', ["ruff check ."]),
        (r"uv run C:\Python311\python.exe -m pytest -q", ["pytest -q"]),
        (r'uv run "C:\Program Files\Python311\python.exe" -m ruff check .', ["ruff check ."]),
        (r"C:/Python311/python.exe -m pytest -q", ["pytest -q"]),
        (r'"C:/Program Files/Python311/python.exe" -m ruff check .', ["ruff check ."]),
        ("python -m ruff check src/app.py", ["ruff check ."]),
        ("cargo test redirect --quiet", ["cargo test"]),
        ("cargo check --workspace", ["cargo check"]),
        ("npm test -- redirect", ["npm test"]),
        ("pnpm test -- redirect", ["pnpm test"]),
        ("go test ./pkg/...", ["go test ./..."]),
        ("ruff check src/app.py", ["ruff check ."]),
        ("make verify TARGET=unit", ["make verify"]),
    ],
)
def test_shell_verification_detection_accepts_exact_and_wrapped_effective_commands(
    cmd: str,
    known: list[str],
) -> None:
    assert _shell_command_is_verification_attempt(cmd, known_verification_commands=known) is True


@pytest.mark.parametrize(
    ("observed", "effective", "expected"),
    [
        ("pytest tests/test_cli.py -q", ["pytest -q", "ruff check ."], {"pytest -q"}),
        ("pytest tests/test_cli.py -v", ["pytest -q", "ruff check ."], {"pytest -q"}),
        ("python -m pytest tests/test_cli.py -v", ["pytest -q", "ruff check ."], {"pytest -q"}),
        (
            "cd /tmp/x && python -m pytest tests/test_batching.py -v",
            ["pytest -q", "ruff check ."],
            {"pytest -q"},
        ),
        ("ruff check src/app.py", ["pytest -q", "ruff check ."], {"ruff check ."}),
        ("cargo test redirect --quiet", ["cargo test", "ruff check ."], {"cargo test"}),
        ("go test ./pkg/...", ["go test ./...", "ruff check ."], {"go test ./..."}),
        ("echo ok", ["pytest -q", "ruff check ."], set()),
    ],
)
def test_matching_effective_verification_commands_maps_targeted_commands_to_coverage(
    observed: str,
    effective: list[str],
    expected: set[str],
) -> None:
    assert (
        _matching_effective_verification_commands(
            observed_command=observed,
            effective_verification_commands=effective,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Improve Express with a small but meaningful PR.", "execute"),
        ("Make a worthwhile improvement in Flask.", "execute"),
        ("Pick a candidate change and implement it.", "execute"),
        ("Why is the parser failing? Fix it.", "execute"),
        (
            "Explain the request lifecycle and then implement the missing retry logic.",
            "execute",
        ),
        ("Review this design and implement the chosen approach.", "execute"),
        ("Walk me through the bug and patch it.", "execute"),
        ("Can you explain why tests are failing and then fix the root cause?", "execute"),
        ("Explain the bug before fixing it.", "execute"),
        ("Walk me through the issue before you patch it.", "execute"),
        ("Review this design, then make the change.", "execute"),
        ("Help me understand the failure and resolve it.", "execute"),
        ("Can you explain the issue before implementing the fix?", "execute"),
        ("Review this design and make the required update.", "execute"),
        ("Explain the issue and clean it up.", "execute"),
        ("Help me understand the failure and correct it.", "execute"),
        ("Explain the bug then apply the required fix.", "execute"),
        ("Could you walk me through the issue before making the change?", "execute"),
        ("Give me a plan only; do not modify files.", "plan_or_analysis_only"),
        ("Review the repo and suggest improvements; no code changes.", "advisory_non_execution"),
        ("How does the parser module work?", "advisory_non_execution"),
        ("What does this module do?", "advisory_non_execution"),
        ("What is the right fix for this parser bug?", "advisory_non_execution"),
        ("What change would you make here?", "advisory_non_execution"),
        ("How should we patch this bug?", "advisory_non_execution"),
        ("What is the safest way to implement this retry?", "advisory_non_execution"),
        ("How would you implement the missing retry logic?", "advisory_non_execution"),
        ("Explain the request lifecycle in this repo.", "advisory_non_execution"),
        ("Why is the parser failing?", "advisory_non_execution"),
        ("Walk me through the request lifecycle in this repo.", "advisory_non_execution"),
        ("Help me understand the parser module.", "advisory_non_execution"),
        ("Review this design", "advisory_non_execution"),
        ("Review this design.", "advisory_non_execution"),
        ("Explain the bug, no code changes.", "advisory_non_execution"),
        ("Just explain how this module works.", "advisory_non_execution"),
        ("Can you implement the parser fix?", "execute"),
        ("Can you improve the repo startup flow?", "execute"),
        ("Πρόσθεσε ένα search command.", "execute"),
        ("Υλοποίησε αυτή την αλλαγή.", "execute"),
        ("Διόρθωσε το bug.", "execute"),
        ("Ενημέρωσε το README και τα tests.", "execute"),
        ("Δώσε μόνο πλάνο.", "plan_or_analysis_only"),
        ("Κάνε μόνο ανάλυση.", "plan_or_analysis_only"),
        ("Μόνο εξήγηση, όχι αλλαγές.", "advisory_non_execution"),
        ("Μην αλλάξεις αρχεία.", "advisory_non_execution"),
        ("Χωρίς αλλαγές στον κώδικα.", "advisory_non_execution"),
        ("Χωρίς να τροποποιήσεις τίποτα.", "advisory_non_execution"),
        ("Τι αλλαγή θα έκανες εδώ;", "advisory_non_execution"),
        ("Ποιος είναι ο καλύτερος τρόπος να διορθώσουμε αυτό το bug;", "advisory_non_execution"),
    ],
)
def test_classify_one_shot_repo_turn_intent(instruction: str, expected: str) -> None:
    assert _classify_one_shot_repo_turn_intent(instruction) == expected


def test_one_shot_marker_matching_is_accent_insensitive_for_greek() -> None:
    assert _normalize_marker_text("Δώσε   μόνο   πλάνο") == _normalize_marker_text(
        "δωσε μονο πλανο"
    )
    assert _classify_one_shot_repo_turn_intent(
        "Δώσε μόνο πλάνο"
    ) == _classify_one_shot_repo_turn_intent("δωσε μονο πλανο")

    progress_accented = "Θα προχωρήσω στην υλοποίηση. Στη συνέχεια θα ενημερώσω το README."
    progress_unaccented = "θα προχωρησω στην υλοποιηση. στη συνεχεια θα ενημερωσω το readme."
    assert _assistant_text_contains_progress_intent(progress_accented) is True
    assert _assistant_text_contains_progress_intent(progress_unaccented) is True

    completion_accented = "Υλοποίησα το search command και έτρεξα τα tests."
    completion_unaccented = "υλοποιησα το search command και ετρεξα τα tests."
    assert _assistant_text_has_completion_marker(completion_accented) is True
    assert _assistant_text_has_completion_marker(completion_unaccented) is True

    blocker_accented = "Δεν μπορώ να προχωρήσω, χρειάζομαι έγκριση."
    blocker_unaccented = "δεν μπορω να προχωρησω, χρειαζομαι εγκριση."
    assert _assistant_text_has_blocker_marker(blocker_accented) is True
    assert _assistant_text_has_blocker_marker(blocker_unaccented) is True


def test_one_shot_continues_after_non_final_progress_message(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-follow-through",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_list", arguments={"path": "."})],
                raw={},
            ),
            LLMResponse(
                content="I will now implement search. Next I will proceed.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_write",
                        arguments={"path": "out.txt", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    assert session.client.calls == 5  # type: ignore[attr-defined]

    events = list(read_session_events(sessions_dir / "one-shot-follow-through.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_non_final_progress_detected" in event_types
    assert "continuation_nudge" in event_types
    assert "one_shot_completion_gate_incomplete_after_retries" not in event_types


def test_one_shot_continues_after_greek_non_final_progress_message(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-follow-through-greek",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_list", arguments={"path": "."})],
                raw={},
            ),
            LLMResponse(
                content="Θα προχωρήσω στην υλοποίηση. Στη συνέχεια θα ενημερώσω το README.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_write",
                        arguments={"path": "out.txt", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Υλοποίησα το search command, ενημέρωσα το README και έτρεξα τα tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Υλοποίησε το search command και ενημέρωσε τα tests.")
    finally:
        session.close()

    assert exit_code == 0
    assert session.client.calls == 5  # type: ignore[attr-defined]

    events = list(read_session_events(sessions_dir / "one-shot-follow-through-greek.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_non_final_progress_detected" in event_types
    assert "continuation_nudge" in event_types
    assert "one_shot_completion_gate_incomplete_after_retries" not in event_types


def test_one_shot_greek_blocker_text_is_not_treated_as_non_final_progress(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-greek-blocker",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="Δεν μπορώ να προχωρήσω, χρειάζομαι έγκριση.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Διόρθωσε αυτό το bug στο repo.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-greek-blocker.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_non_final_progress_detected" not in event_types
    assert "continuation_nudge" not in event_types


def test_one_shot_greek_completion_text_is_not_treated_as_non_final_progress(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-greek-completion-no-evidence",
    )
    completion_text = "Υλοποίησα το search command, πρόσθεσα tests και έτρεξα τα tests."
    session.client = _ScriptedClient(
        [
            LLMResponse(content=completion_text, tool_calls=[], raw={}),
            LLMResponse(content=completion_text, tool_calls=[], raw={}),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Υλοποίησε το search command και ενημέρωσε τα tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-greek-completion-no-evidence.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_non_final_progress_detected" not in event_types
    assert "continuation_nudge" not in event_types
    assert "one_shot_completion_gate_incomplete_after_retries" in event_types


def test_one_shot_final_completion_response_finishes_normally(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-final",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "final.txt", "content": "ready\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    assert session.client.calls == 3  # type: ignore[attr-defined]

    events = list(read_session_events(sessions_dir / "one-shot-final.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_non_final_progress_detected" not in event_types
    assert "continuation_nudge" not in event_types
    assert "one_shot_completion_gate_incomplete_after_retries" not in event_types


def test_one_shot_plan_only_request_does_not_auto_continue(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-plan-only",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="Plan: 1) inspect files 2) add command 3) test.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Give me a plan only; do not modify files.")
    finally:
        session.close()

    assert exit_code == 0
    assert session.client.calls == 1  # type: ignore[attr-defined]

    events = list(read_session_events(sessions_dir / "one-shot-plan-only.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "continuation_nudge" not in event_types
    assert "one_shot_non_final_progress_detected" not in event_types


def test_one_shot_review_only_request_is_exempt_from_execution_safeguards(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-review-only",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="Suggested improvements: tighten verification, reduce exploration loops, and improve final reporting.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Review the repo and suggest improvements; no code changes.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-review-only.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_non_final_progress_detected" not in event_types
    assert "continuation_nudge" not in event_types
    assert "one_shot_completion_gate_failed" not in event_types


@pytest.mark.parametrize(
    ("instruction", "response", "session_id"),
    [
        (
            "How does the parser module work?",
            "The parser normalizes CLI input, validates flags, and then hands structured options to the request pipeline.",
            "one-shot-explanatory-repo-question",
        ),
        (
            "What is the right fix for this parser bug?",
            "The safest fix is to tighten flag normalization before option validation and add a focused parser regression test.",
            "one-shot-hypothetical-fix-question",
        ),
        (
            "How would you implement the missing retry logic?",
            "I would add a retry budget in the request loop, thread it through the parser options, and cover it with a focused verification test.",
            "one-shot-hypothetical-implementation-question",
        ),
    ],
)
def test_one_shot_non_execution_repo_question_is_exempt_from_execution_safeguards(
    tmp_path: Path,
    instruction: str,
    response: str,
    session_id: str,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content=response,
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn(instruction)
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_non_final_progress_detected" not in event_types
    assert "continuation_nudge" not in event_types
    assert "one_shot_completion_gate_failed" not in event_types


@pytest.mark.parametrize(
    ("instruction", "session_id"),
    [
        ("Implement search command.", "one-shot-repeat-progress"),
        ("Make a worthwhile improvement in Flask.", "one-shot-repeat-progress-flask"),
    ],
)
def test_one_shot_non_final_progress_stopped_repeated_progress_emits_forced_final_summary(
    tmp_path: Path,
    instruction: str,
    session_id: str,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
        surface=surface,
    )
    repeated_progress_text = "I will implement search next."
    client = _ScriptedClient(
        [
            LLMResponse(
                content=repeated_progress_text,
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content=repeated_progress_text,
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content=(
                    "Completed work: identified the requested implementation step.\n"
                    "Remaining work: the change is still incomplete.\n"
                    "Known issues or risks: repeated non-final progress stopped the run."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn(instruction)
    finally:
        session.close()

    assert exit_code == 1
    assert client.calls == 3

    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    assert any(event.get("type") == "continuation_nudge" for event in events)
    incomplete_events = [
        event for event in events if event.get("type") == "one_shot_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("reason") == "repeated_progress"
    expected_error = (
        "One-shot run stopped: model returned repeated/non-final progress text "
        "without continuing implementation."
    )
    assert surface.errors
    assert surface.errors[-1] == expected_error
    requested = [event for event in events if event.get("type") == "forced_final_summary_requested"]
    assert requested
    assert dict(requested[-1].get("payload") or {}).get("reason") == (
        "non_final_progress_retry_exhausted"
    )
    summary = _assert_forced_final_summary_emitted(events)
    assert surface.final_messages[-1] == summary
    _assert_last_forced_summary_request(
        client,
        latest_assistant_text=repeated_progress_text,
        termination_cause="repeated non-final progress is detected",
    )


def test_forced_final_summary_falls_back_when_model_returns_tool_markup(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-forced-summary-tool-markup",
        surface=surface,
    )
    progress_text = "I will implement search next."
    raw_tool_markup = (
        "<｜｜DSML｜｜tool_calls>\n"
        '<｜｜DSML｜｜invoke name="shell_run">\n'
        '<｜｜DSML｜｜parameter name="cmd">pytest -q</｜｜DSML｜｜parameter>\n'
        "</｜｜DSML｜｜invoke>\n"
        "</｜｜DSML｜｜tool_calls>"
    )
    client = _ScriptedClient(
        [
            LLMResponse(content=progress_text, tool_calls=[], raw={}),
            LLMResponse(content=progress_text, tool_calls=[], raw={}),
            LLMResponse(content=raw_tool_markup, tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-forced-summary-tool-markup.jsonl"))
    fallback_events = [
        event for event in events if event.get("type") == "forced_final_summary_fallback"
    ]
    assert fallback_events
    assert dict(fallback_events[-1].get("payload") or {}).get("fallback_reason") == (
        "tool_call_markup_response"
    )
    summary = _assert_forced_final_summary_emitted(events)
    assert "DSML" not in summary
    assert "tool_calls" not in summary
    assert surface.final_messages[-1] == summary


def test_one_shot_non_final_progress_stopped_at_continuation_cap_emits_forced_final_summary(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-continuation-cap-forced-summary",
        surface=surface,
    )
    latest_progress_text = "I will add tests next."
    client = _ScriptedClient(
        [
            LLMResponse(content="I will inspect the parser next.", tool_calls=[], raw={}),
            LLMResponse(content="I will update the parser next.", tool_calls=[], raw={}),
            LLMResponse(content=latest_progress_text, tool_calls=[], raw={}),
            LLMResponse(
                content=(
                    "Completed work: partial implementation progress was reported.\n"
                    "Remaining work: the requested change is still unfinished.\n"
                    "Known issues or risks: the non-final progress continuation cap was reached."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    assert client.calls == 4
    events = list(
        read_session_events(sessions_dir / "one-shot-continuation-cap-forced-summary.jsonl")
    )
    incomplete_events = [
        event for event in events if event.get("type") == "one_shot_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("reason") == "continuation_cap"
    expected_error = (
        "One-shot run stopped: model returned repeated/non-final progress text "
        "without continuing implementation."
    )
    assert surface.errors
    assert surface.errors[-1] == expected_error
    requested = [event for event in events if event.get("type") == "forced_final_summary_requested"]
    assert requested
    assert dict(requested[-1].get("payload") or {}).get("reason") == (
        "non_final_progress_continuation_cap_reached"
    )
    summary = _assert_forced_final_summary_emitted(events)
    assert surface.final_messages[-1] == summary
    _assert_last_forced_summary_request(
        client,
        latest_assistant_text=latest_progress_text,
        termination_cause="the non-final progress continuation limit is reached",
    )


def test_one_shot_completion_gate_rejects_empty_final_response(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-empty-final",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(content="", tool_calls=[], raw={}),
            LLMResponse(content="", tool_calls=[], raw={}),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-empty-final.jsonl"))
    assert any(event.get("type") == "completion_gate_nudge" for event in events)
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "empty_final_response" in problems


def test_interactive_completion_gate_rejects_claim_only_without_evidence(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session_id = "interactive-claim-no-evidence"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
            LLMResponse(
                content=(
                    "Completed work: no repository edits were verified.\n"
                    "Remaining work: implementation still needs to be done.\n"
                    "Known issues or risks: the completion gate stopped a claim-only reply."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "interactive_completion_gate_failed" in event_types
    assert "interactive_completion_gate_incomplete_after_retries" in event_types
    assert "one_shot_completion_gate_failed" not in event_types
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "interactive_completion_gate_incomplete_after_retries"
    ]
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("runtime_kind") == "interactive_chat"
    assert set(payload.get("problems") or []) >= {"no_material_edits", "verification_not_attempted"}
    nudge_events = [event for event in events if event.get("type") == "completion_gate_nudge"]
    assert nudge_events
    assert "interactive execution turn" in str(
        dict(nudge_events[-1].get("payload") or {}).get("message") or ""
    )


@pytest.mark.parametrize(
    ("instruction", "session_id"),
    [
        ("Implement search command and update tests.", "one-shot-claim-no-evidence"),
        (
            "Improve Express with a small but meaningful PR.",
            "one-shot-claim-no-evidence-express",
        ),
        (
            "Why is the parser failing? Fix it.",
            "one-shot-claim-no-evidence-fix-it",
        ),
        (
            "Explain the bug before fixing it.",
            "one-shot-claim-no-evidence-before-fixing",
        ),
        (
            "Walk me through the issue before you patch it.",
            "one-shot-claim-no-evidence-before-you-patch",
        ),
        (
            "Review this design, then make the change.",
            "one-shot-claim-no-evidence-make-the-change",
        ),
        (
            "Help me understand the failure and resolve it.",
            "one-shot-claim-no-evidence-resolve-it",
        ),
        (
            "Can you explain the issue before implementing the fix?",
            "one-shot-claim-no-evidence-implementing-fix",
        ),
        (
            "Review this design and make the required update.",
            "one-shot-claim-no-evidence-required-update",
        ),
        (
            "Could you walk me through the issue before making the change?",
            "one-shot-claim-no-evidence-before-making-change",
        ),
    ],
)
def test_one_shot_completion_gate_rejects_claim_only_without_evidence(
    tmp_path: Path,
    instruction: str,
    session_id: str,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn(instruction)
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "no_material_edits" in problems
    assert "verification_not_attempted" in problems
    assert payload.get("stage") == "no_material_edits"


def test_one_shot_no_material_edits_repo_activity_uses_distinct_stage_and_strong_nudge(
    tmp_path: Path,
) -> None:
    _write_test_files(tmp_path, ["src/mini_notes/logic.py"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-no-material-distinct-stage"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    assert any(event.get("type") == "one_shot_no_material_edits_detected" for event in events)
    assert not any(event.get("type") == "implementation_bootstrap_nudge" for event in events)
    nudge_events = [
        event for event in events if event.get("type") == "no_material_edits_bootstrap_nudge"
    ]
    assert nudge_events
    nudge_payload = dict(nudge_events[-1].get("payload") or {})
    message = str(nudge_payload.get("message") or "")
    assert "Do not finalize or summarize yet." in message
    assert "Your next step must be a real action/progress tool call" in message
    assert "Likely repo-root-relative targets: src/mini_notes/logic.py." in message
    assert nudge_payload.get("repo_tool_activity_observed") is True
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    assert payload.get("stage") == "no_material_edits"


def test_one_shot_no_material_edits_bootstrap_nudge_includes_recent_path_anchors(
    tmp_path: Path,
) -> None:
    _write_test_files(
        tmp_path,
        [
            "src/mini_notes/logic.py",
            "src/mini_notes/cli.py",
            "tests/test_cli.py",
        ],
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-no-material-path-anchors"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_read",
                        arguments={"path": "tests/test_cli.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    nudge_events = [
        event for event in events if event.get("type") == "no_material_edits_bootstrap_nudge"
    ]
    assert nudge_events
    payload = dict(nudge_events[-1].get("payload") or {})
    anchor_paths = list(payload.get("anchor_paths") or [])
    assert "src/mini_notes/logic.py" in anchor_paths
    assert "tests/test_cli.py" in anchor_paths
    message = str(payload.get("message") or "")
    assert "src/mini_notes/logic.py" in message
    assert "tests/test_cli.py" in message


def test_one_shot_no_material_edits_bootstrap_recovers_after_real_action(tmp_path: Path) -> None:
    _write_test_files(tmp_path, ["src/mini_notes/logic.py"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-no-material-recovers"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_write",
                        arguments={"path": "src/mini_notes/logic.py", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    assert any(event.get("type") == "no_material_edits_bootstrap_nudge" for event in events)
    assert not any(
        event.get("type") == "one_shot_no_material_edits_incomplete_after_retries"
        for event in events
    )


def test_one_shot_completion_gate_terminal_failure_no_material_edits_emits_forced_final_summary(
    tmp_path: Path,
) -> None:
    _write_test_files(tmp_path, ["src/mini_notes/logic.py"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-no-material-fails-after-cap"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
        surface=surface,
    )
    latest_final_text = "Implemented search, updated README, and ran tests."
    client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content=latest_final_text,
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content=latest_final_text,
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content=(
                    "Completed work: repo activity was observed and verification ran.\n"
                    "Remaining work: no material code edits were completed.\n"
                    "Known issues or risks: completion-gate repair attempts were exhausted."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / f"{session_id}.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_no_material_edits_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("stage") == "no_material_edits"
    assert payload.get("repo_tool_activity_observed") is True
    assert "src/mini_notes/logic.py" in list(payload.get("anchor_paths") or [])
    assert any(
        event.get("type") == "one_shot_completion_gate_incomplete_after_retries" for event in events
    )
    expected_error = (
        "One-shot run stopped: completion gate requirements were not met (no material edits)."
    )
    assert surface.errors
    assert surface.errors[-1] == expected_error
    requested = [event for event in events if event.get("type") == "forced_final_summary_requested"]
    assert requested
    assert dict(requested[-1].get("payload") or {}).get("reason") == (
        "completion_gate_terminal_failure"
    )
    summary = _assert_forced_final_summary_emitted(events)
    assert surface.final_messages[-1] == summary
    _assert_last_forced_summary_request(
        client,
        latest_assistant_text=latest_final_text,
        termination_cause="completion-gate repair attempts are exhausted",
    )


def test_one_shot_completion_gate_rejects_failing_verification(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verification-failed",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "target.txt", "content": "implemented\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_FAIL_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-verification-failed.jsonl"))
    assert any(event.get("type") == "completion_gate_nudge" for event in events)
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


def test_one_shot_blocker_text_does_not_bypass_code_verification_failure(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=[_VERIFY_FAIL_COMMAND],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-code-failure-not-blocker",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('changed')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_FAIL_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I am blocked because verification is failing.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="I am blocked because verification is failing.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-code-failure-not-blocker.jsonl"))
    failed_events = [
        event for event in events if event.get("type") == "one_shot_completion_gate_failed"
    ]
    assert failed_events
    payload = dict(failed_events[-1].get("payload") or {})
    assert payload.get("blocked_response") is True
    assert payload.get("blocked_response_allows_completion") is False
    assert "verification_failed" in set(payload.get("problems") or [])


def test_one_shot_infra_verification_blocker_can_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_task_verification(
        *,
        root: Path,
        commands: list[str],
        artifact_path: Path,
        cfg: AppConfig,
    ) -> VerifyRunResult:
        _ = root, cfg
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            "$ ./gradlew test\n/bin/bash: ./gradlew: No such file or directory\n",
            encoding="utf-8",
        )
        return VerifyRunResult(
            commands=list(commands),
            command_results=[
                VerifyCommandResult(
                    command="./gradlew test",
                    exit_code=127,
                    output="/bin/bash: ./gradlew: No such file or directory\n",
                    stderr="/bin/bash: ./gradlew: No such file or directory\n",
                    real_execution=False,
                    non_execution_reason="execution_layer_failure",
                )
            ],
            artifact_path=artifact_path,
            failure_category="infra_unavailable",
        )

    monkeypatch.setattr(agent_loop_mod, "run_task_verification", fake_run_task_verification)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["./gradlew test"],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-infra-blocker-finalizes",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('changed')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="verify_run", arguments={})],
                raw={},
            ),
            LLMResponse(
                content=(
                    "I am blocked because `./gradlew` is missing, so the configured "
                    "verification command cannot run."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-infra-blocker-finalizes.jsonl"))
    assert not [event for event in events if event.get("type") == "one_shot_completion_gate_failed"]


def test_one_shot_completion_gate_allows_failed_verification_repair_cycle_to_recover(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-repair-recover",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the change.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_IMPORT_ERROR_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the change and verified it.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('repaired')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and verification now passes.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-verify-repair-recover.jsonl"))
    gate_failed_events = [
        event for event in events if event.get("type") == "one_shot_completion_gate_failed"
    ]
    assert gate_failed_events
    stages = [str((event.get("payload") or {}).get("stage") or "") for event in gate_failed_events]
    assert "verification_not_attempted" in stages
    assert "verification_failed" in stages
    assert any(event.get("type") == "failed_verification_repair_attempt" for event in events)
    assert not any(
        event.get("type") == "one_shot_completion_gate_incomplete_after_retries" for event in events
    )


def test_one_shot_completion_gate_terminal_failure_after_failed_verification_repair_emits_forced_final_summary(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-repair-terminal",
        surface=surface,
    )
    latest_final_text = "Implemented and verified successfully."
    client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_NAME_ERROR_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content=latest_final_text,
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('repair attempt')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_NAME_ERROR_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content=latest_final_text,
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content=(
                    "Completed work: retried the failing verification repair path.\n"
                    "Remaining work: verification is still failing and needs another fix.\n"
                    "Known issues or risks: completion-gate repair attempts were exhausted."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-verify-repair-terminal.jsonl"))
    assert any(event.get("type") == "failed_verification_repair_attempt" for event in events)
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("stage") == "verification_failed"
    assert payload.get("problem_summary") == "verification failing"
    assert "NameError" in str(payload.get("verification_failure_snippet") or "")
    expected_error = (
        "One-shot run stopped: completion gate requirements were not met "
        "(verification failing). First reported error: NameError: search_notes is not defined."
    )
    assert surface.errors
    assert surface.errors[-1] == expected_error
    requested = [event for event in events if event.get("type") == "forced_final_summary_requested"]
    assert requested
    assert dict(requested[-1].get("payload") or {}).get("reason") == (
        "completion_gate_terminal_failure"
    )
    summary = _assert_forced_final_summary_emitted(events)
    assert surface.final_messages[-1] == summary
    _assert_last_forced_summary_request(
        client,
        latest_assistant_text=latest_final_text,
        termination_cause="completion-gate repair attempts are exhausted",
    )


def test_one_shot_completion_gate_nudge_includes_first_failed_verification_snippet(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=10,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-snippet-nudge",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_IMPORT_ERROR_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented and verified successfully.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented and verified successfully.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-verify-snippet-nudge.jsonl"))
    nudge_events = [event for event in events if event.get("type") == "completion_gate_nudge"]
    assert nudge_events
    nudge_payload = dict(nudge_events[-1].get("payload") or {})
    assert nudge_payload.get("stage") == "verification_failed"
    snippet = str(nudge_payload.get("verification_failure_snippet") or "")
    message = str(nudge_payload.get("message") or "")
    assert "ImportError" in snippet
    assert "search_notes" in snippet
    assert "First reported error:" in message
    assert "ImportError" in message


def test_one_shot_final_summary_is_rewritten_to_greek_with_explicit_override(
    tmp_path: Path,
) -> None:
    _write_test_files(tmp_path, ["src/app.py"])

    cfg = AppConfig(model="test-model", routing_mode="auto")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-final-rewrite-greek",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content=(
                    "Implemented `search_notes` in `src/app.py` and ran "
                    "`pytest tests/test_cli.py -q`."
                ),
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content=(
                    "Υλοποίησα το `search_notes` στο `src/app.py` και έτρεξα "
                    "`pytest tests/test_cli.py -q`."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]
    session.router_client = _ScriptedClient(
        [
            LLMResponse(
                content=(
                    '{"route":"repo","execution_posture":"execute","confidence":0.99,'
                    '"reply":"","language":"Greek","script":"Greek",'
                    '"explicit_language_override":true,"tool_family":"none",'
                    '"tool_candidates":[]}'
                ),
                tool_calls=[],
                raw={},
            )
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn(
            "Απάντησε στα Ελληνικά για το τελικό summary και υλοποίησε την αλλαγή."
        )
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-final-rewrite-greek.jsonl"))
    rewrite_events = [event for event in events if event.get("type") == "final_summary_rewrite"]
    assert rewrite_events
    rewrite_payload = dict(rewrite_events[-1].get("payload") or {})
    assert rewrite_payload.get("status") == "applied"
    final_events = [event for event in events if event.get("type") == "final"]
    assert final_events
    final_content = str((final_events[-1].get("payload") or {}).get("content") or "")
    assert final_content.startswith("Υλοποίησα")
    assert "`search_notes`" in final_content
    assert "`src/app.py`" in final_content
    assert "`pytest tests/test_cli.py -q`" in final_content


def test_final_summary_rewrite_keeps_original_when_technical_tokens_change() -> None:
    original = (
        "Implemented `search_notes` in `src/mini_notes/logic.py` and updated "
        "`verify_commands`.\n\n```bash\npython -m pytest -q\n```"
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content=(
                    "Υλοποίησα το `search_notes` στο `src/mini_notes/logic.py`.\n\n"
                    "```bash\npytest -q\n```"
                ),
                tool_calls=[],
                raw={},
            )
        ]
    )

    rewritten, payload = _rewrite_final_summary_for_language(
        client=client,
        final_text=original,
        language="Greek",
        script="",
        explicit_language_override=True,
    )

    assert rewritten == original
    assert payload is not None
    assert payload.get("status") == "kept_original"
    assert payload.get("reason") == "protected_tokens_missing"


def test_one_shot_runtime_messages_remain_english_without_explicit_override(
    tmp_path: Path,
) -> None:
    _write_test_files(tmp_path, ["src/app.py"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-runtime-english-default",
        surface=surface,
    )
    final_summary = "Implemented the change and ran `pytest tests/test_cli.py -q`."
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(content=final_summary, tool_calls=[], raw={}),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the change and verify it.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-runtime-english-default.jsonl"))
    progress_events = [
        str((event.get("payload") or {}).get("message") or "")
        for event in events
        if event.get("type") == "progress"
    ]
    assert "Understanding your request." in "\n".join(progress_events)
    assert all("Κατανοώ το αίτημά σου." not in message for message in progress_events)
    assert not any(event.get("type") == "final_summary_rewrite" for event in events)
    assert surface.final_messages[-1] == final_summary


def test_runtime_message_falls_back_to_english_for_unsupported_language() -> None:
    assert (
        _runtime_message(
            "phase_understanding_request",
            language="Spanish",
            explicit_language_override=True,
        )
        == "Understanding your request."
    )

    client = _ScriptedClient(
        [
            LLMResponse(
                content="Implementé `search_notes` en `src/mini_notes/logic.py`.",
                tool_calls=[],
                raw={},
            )
        ]
    )
    rewritten, payload = _rewrite_final_summary_for_language(
        client=client,
        final_text="Implemented `search_notes` in `src/mini_notes/logic.py`.",
        language="Spanish",
        script="",
        explicit_language_override=True,
    )

    assert rewritten.startswith("Implementé")
    assert "`search_notes`" in rewritten
    assert "`src/mini_notes/logic.py`" in rewritten
    assert payload is not None
    assert payload.get("status") == "applied"


def test_one_shot_docs_only_readme_change_does_not_require_verification(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-docs-readme-only",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "README.md", "content": "# Setup\n\nInstall docs.\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Updated README.md with installation instructions.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Update README.md with installation instructions.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-docs-readme-only.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_completion_gate_failed" not in event_types


def test_one_shot_docs_only_readme_change_requires_explicit_verification_contract() -> None:
    assert _verification_expected_for_turn(
        turn_intent="execute",
        blocked=False,
        touched_repo_paths={"README.md"},
        verification_contract_requires_execution=True,
    )
    assert _verification_expected_for_turn(
        turn_intent="execute",
        blocked=True,
        touched_repo_paths={"README.md"},
        verification_contract_requires_execution=True,
    )
    assert not _verification_expected_for_turn(
        turn_intent="execute",
        blocked=False,
        touched_repo_paths={"README.md"},
        verification_contract_requires_execution=False,
    )


def test_one_shot_docs_only_docs_dir_change_does_not_require_verification(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-docs-dir-only",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "docs/install.md",
                            "content": "# Install\n\nDocumented setup.\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Documented installation steps under docs/.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Document installation steps under docs/.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-docs-dir-only.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_completion_gate_failed" not in event_types


def test_one_shot_source_code_change_without_verification_still_fails(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-code-no-verify",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-code-no-verify.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_not_attempted" in problems


def test_one_shot_mixed_docs_and_code_change_without_verification_still_fails(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-mixed-docs-code-no-verify",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "README.md", "content": "# Setup\n\nUpdated docs.\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and updated docs.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and updated docs.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change and update docs.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-mixed-docs-code-no-verify.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_not_attempted" in problems


def test_one_shot_source_code_change_with_passing_verification_succeeds(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-code-verify-ok",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and verified it.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-code-verify-ok.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_completion_gate_incomplete_after_retries" not in event_types


def test_one_shot_inferred_make_verify_shell_run_counts_as_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "Makefile").write_text("verify:\n\t@echo ok\n", encoding="utf-8")

    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-make-verify-shell",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": "make verify"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran make verify.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        env_message = next(
            (
                str(message.get("content") or "")
                for message in session.messages
                if message.get("role") == "user"
                and "<environment_context>" in str(message.get("content") or "")
            ),
            "",
        )
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert 'recommended_verification_commands: ["make verify"]' in env_message
    assert exit_code == 0


def test_one_shot_inferred_just_verify_shell_run_counts_as_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "justfile").write_text("verify:\n    @echo ok\n", encoding="utf-8")

    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-just-verify-shell",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": "just verify"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran just verify.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        env_message = next(
            (
                str(message.get("content") or "")
                for message in session.messages
                if message.get("role") == "user"
                and "<environment_context>" in str(message.get("content") or "")
            ),
            "",
        )
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert 'recommended_verification_commands: ["just verify"]' in env_message
    assert exit_code == 0


def test_one_shot_authoritative_shell_run_counts_as_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-authoritative-shell",
        authoritative_verification_commands=["PYTHONPATH=src pytest -q"],
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": "PYTHONPATH=src pytest -q"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran the authoritative verification command.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_config_fallback_shell_run_counts_as_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["python -m pytest -q"],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-config-fallback-shell",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": "python -m pytest -q"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran the configured verification command.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


@pytest.mark.parametrize(
    ("shell_cmd", "session_suffix"),
    [
        ("echo pytest -q", "echo-pytest"),
        ("pytest -q || true", "pytest-or-true"),
        ("bash -lc 'pytest -q || true'", "wrapped-pytest-or-true"),
    ],
)
def test_one_shot_shell_run_that_is_not_safe_verification_does_not_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shell_cmd: str,
    session_suffix: str,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "exit_code": 0, "stdout": "pytest -q\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q"],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=f"one-shot-{session_suffix}-no-verify",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": shell_cmd},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / f"one-shot-{session_suffix}-no-verify.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_not_attempted" in problems


def test_one_shot_verify_run_rejects_unsafe_compound_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for unsafe commands"),
    )

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q"],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-run-unsafe-command",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": ["pytest -q || true"]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-verify-run-unsafe-command.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


def test_one_shot_verify_run_incompatible_override_does_not_count(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q"],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-run-incompatible-override",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": ['python3 -c "print(123)"']},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(
        read_session_events(sessions_dir / "one-shot-verify-run-incompatible-override.jsonl")
    )
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


def test_one_shot_verify_run_go_no_tests_to_run_does_not_count(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["go test ./..."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-run-go-no-tests",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "main.go", "content": "package main\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_GO_NO_TESTS_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-verify-run-go-no-tests.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


def test_one_shot_verify_run_go_no_test_files_does_not_count(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["go test ./..."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-run-go-no-test-files",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "main.go", "content": "package main\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_GO_NO_TEST_FILES_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-verify-run-go-no-test-files.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


def test_one_shot_partial_shell_verification_coverage_does_not_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "effective_cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-partial-shell-verification-coverage",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(
        read_session_events(sessions_dir / "one-shot-partial-shell-verification-coverage.jsonl")
    )
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert problems & {"verification_incomplete", "verification_failed"}
    assert "ruff check ." in set(payload.get("missing_verification_commands") or [])


def test_one_shot_partial_verify_run_coverage_does_not_count(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-partial-verify-run-coverage",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc2", name="verify_run", arguments={"commands": ["pytest -q"]})
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-partial-verify-run-coverage.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert problems & {"verification_incomplete", "verification_failed"}
    assert "ruff check ." in set(payload.get("missing_verification_commands") or [])


def test_one_shot_two_shell_verification_commands_cover_effective_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "effective_cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-full-shell-verification-coverage",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc3", name="shell_run", arguments={"cmd": "ruff check ."})
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_targeted_verify_run_commands_cover_effective_contract(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-targeted-verify-run-coverage",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={
                            "commands": [
                                "pytest tests/test_cli.py -q",
                                "ruff check src/app.py",
                            ]
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_verify_run_coverage_becomes_stale_after_later_code_edit(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-verify-run-stale-after-code-edit"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('first')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('second')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_incomplete" in problems
    assert payload.get("verification_coverage_stale") is True
    assert "pytest -q" in set(payload.get("missing_verification_commands") or [])


def test_one_shot_shell_verification_coverage_becomes_stale_after_later_code_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "effective_cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-shell-verify-stale-after-code-edit"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('first')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('second')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_incomplete" in problems
    assert payload.get("verification_coverage_stale") is True
    assert "pytest -q" in set(payload.get("missing_verification_commands") or [])


def test_one_shot_shell_edit_after_verification_invalidates_freshness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate_cmd = "python3 -c 'rewrite src/app.py'"
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(
            mutations={mutate_cmd: ("src/app.py", "print('shell changed')\n")}
        ),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-shell-edit-invalidates-verification"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="shell_run", arguments={"cmd": mutate_cmd})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_incomplete" in problems
    assert payload.get("verification_coverage_stale") is True
    assert "pytest -q" in set(payload.get("missing_verification_commands") or [])


def test_one_shot_shell_only_repo_edit_counts_as_material_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate_cmd = "python3 -c 'write src/app.py'"
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(
            mutations={mutate_cmd: ("src/app.py", "print('shell edit')\n")}
        ),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-shell-only-material-edit",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="shell_run", arguments={"cmd": mutate_cmd})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_shell_docs_only_edit_after_verify_keeps_verification_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate_cmd = "python3 -c 'write README.md'"
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(
            mutations={mutate_cmd: ("README.md", "# Docs\n\nUpdated via shell.\n")}
        ),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-shell-docs-edit-after-verify",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="shell_run", arguments={"cmd": mutate_cmd})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and updated docs.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_shell_verification_cache_artifacts_do_not_count_as_material_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(mutations={}),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-shell-verify-cache-not-material"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="Ran verification successfully.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Ran verification successfully.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "no_material_edits" in problems


def test_one_shot_shell_verification_that_mutates_code_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(
            mutations={"pytest -q": ("src/app.py", "print('mutated during verify')\n")}
        ),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-shell-verify-mutates-code"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


def test_one_shot_verify_run_that_mutates_code_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        _verify_run_with_repo_mutations(
            mutations_by_command={
                _VERIFY_OK_COMMAND: [("src/app.py", "print('mutated during verify')\n")]
            }
        ),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-verify-run-mutates-code"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems
    state = dict(payload.get("state") or {})
    assert "verify_run" in set(state.get("material_edit_tools") or [])


def test_one_shot_failed_shell_edit_after_verification_invalidates_freshness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate_cmd = "python3 -c 'rewrite src/app.py and fail'"
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(
            mutations={mutate_cmd: ("src/app.py", "print('shell changed after verify')\n")},
            exit_codes={mutate_cmd: 1},
            stderrs={mutate_cmd: "command failed\n"},
        ),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-failed-shell-edit-stales-verification"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="shell_run", arguments={"cmd": mutate_cmd})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_incomplete" in problems
    assert payload.get("verification_coverage_stale") is True
    assert "pytest -q" in set(payload.get("missing_verification_commands") or [])


def test_one_shot_failed_shell_only_repo_edit_counts_as_material_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate_cmd = "python3 -c 'write src/app.py then fail'"
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(
            mutations={mutate_cmd: ("src/app.py", "print('shell edit before verify')\n")},
            exit_codes={mutate_cmd: 1},
            stderrs={mutate_cmd: "command failed\n"},
        ),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-failed-shell-material-edit",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="shell_run", arguments={"cmd": mutate_cmd})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="shell_run", arguments={"cmd": "pytest -q"})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_verify_run_docs_only_edit_after_verify_keeps_verification_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        _verify_run_with_repo_mutations(
            mutations_by_call={2: [("README.md", "# Docs\n\nUpdated via verify.\n")]}
        ),
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-run-docs-edit-after-verify",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and updated docs.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_docs_only_edit_after_verify_does_not_invalidate_coverage(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=7,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-docs-edit-after-verify-fresh",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('first')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "README.md", "content": "# Docs\n\nUpdated.\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and updated docs.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_code_edit_after_verify_requires_rerunning_verification(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only", verify_commands=["pytest -q"])
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-reverify-after-code-edit",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('first')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('second')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and reran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_multicommand_verification_becomes_stale_after_code_edit(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-multicommand-stale-after-code-edit"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('first')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND, _VERIFY_RUFF_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('second')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_incomplete" in problems
    assert payload.get("verification_coverage_stale") is True
    assert set(payload.get("missing_verification_commands") or []) == {
        "pytest -q",
        "ruff check .",
    }


def test_one_shot_multicommand_shell_code_edit_invalidates_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate_cmd = "python3 -c 'rewrite src/app.py'"
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(
            mutations={mutate_cmd: ("src/app.py", "print('shell changed')\n")}
        ),
    )

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-multicommand-shell-edit-stale"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND, _VERIFY_RUFF_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="shell_run", arguments={"cmd": mutate_cmd})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_incomplete" in problems
    assert payload.get("verification_coverage_stale") is True
    assert set(payload.get("missing_verification_commands") or []) == {
        "pytest -q",
        "ruff check .",
    }


def test_one_shot_multicommand_docs_only_edit_keeps_verification_fresh(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-multicommand-docs-edit-after-verify",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('first')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND, _VERIFY_RUFF_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "docs/notes.md", "content": "# Notes\n\nUpdated.\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and updated docs.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_multicommand_shell_docs_edit_keeps_verification_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate_cmd = "python3 -c 'rewrite README.md'"
    monkeypatch.setattr(
        agent_loop_mod,
        "shell_run",
        _shell_run_with_repo_mutations(
            mutations={mutate_cmd: ("README.md", "# Docs\n\nUpdated via shell.\n")}
        ),
    )

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-multicommand-shell-docs-edit",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('initial')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND, _VERIFY_RUFF_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="shell_run", arguments={"cmd": mutate_cmd})],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and updated docs.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_multicommand_rerunning_only_one_command_after_code_edit_is_incomplete(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session_id = "one-shot-multicommand-rerun-one-after-edit"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=9,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('first')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND, _VERIFY_RUFF_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('second')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and reran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and reran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    payload = _latest_completion_gate_payload(sessions_dir, session_id)
    problems = set(payload.get("problems") or [])
    assert "verification_incomplete" in problems
    assert payload.get("verification_coverage_stale") is False
    assert set(payload.get("missing_verification_commands") or []) == {"ruff check ."}


def test_one_shot_multicommand_rerunning_all_commands_after_code_edit_passes(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q", "ruff check ."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=9,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-multicommand-rerun-all-after-edit",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('first')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND, _VERIFY_RUFF_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('second')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND, _VERIFY_RUFF_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and reran all verification commands.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


@pytest.mark.parametrize(
    ("verify_commands", "override_command", "session_suffix"),
    [
        (["cargo test"], "cargo test --no-run", "cargo-no-run"),
        (["cargo test"], "cargo test -- --list", "cargo-forwarded-list"),
        (["pytest -q"], "pytest -q --setup-plan", "pytest-setup-plan"),
        (["pytest -q"], "pytest -q --co", "pytest-collect-only-alias"),
        (["go test ./..."], "go test -c ./...", "go-compile-only"),
        (["go test ./..."], "go test -run '^$' ./...", "go-zero-run"),
    ],
)
def test_one_shot_verify_run_non_executing_override_does_not_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verify_commands: list[str],
    override_command: str,
    session_suffix: str,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail(
            "verify engine should not run for non-executing verification overrides"
        ),
    )

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=verify_commands,
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=f"one-shot-verify-run-non-executing-{session_suffix}",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [override_command]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(
        read_session_events(
            sessions_dir / f"one-shot-verify-run-non-executing-{session_suffix}.jsonl"
        )
    )
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


@pytest.mark.parametrize(
    ("verify_commands", "shell_cmd", "session_suffix"),
    [
        (["pytest -q"], "bash -lc 'pytest -q'", "wrapped-pytest"),
        (["pytest -q"], "python -m pytest -q", "python-module-pytest"),
        (["pytest -q"], "env PYTHONPATH=src pytest -q", "env-pytest"),
        (["pytest -q"], "poetry run pytest -q", "poetry-pytest"),
        (["pytest -q"], "pytest tests/test_cli.py -q", "targeted-pytest"),
        (["pytest -q"], "pytest tests/test_cli.py -v", "targeted-pytest-verbose"),
        (["pytest -q"], "python -m pytest tests/test_cli.py -q", "targeted-python-module-pytest"),
        (["cargo test"], "cargo test redirect --quiet", "cargo-targeted-test"),
        (["go test ./..."], _VERIFY_GO_OK_COMMAND, "go-targeted-test"),
        (["go test ./..."], _VERIFY_GO_MIXED_OK_COMMAND, "go-mixed-targeted-test"),
        (["npm test"], "npm test -- redirect", "npm-targeted-test"),
    ],
)
def test_one_shot_equivalent_real_verification_commands_still_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verify_commands: list[str],
    shell_cmd: str,
    session_suffix: str,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {"cmd": cmd, "exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=verify_commands,
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override=f"one-shot-equivalent-{session_suffix}-verify",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": shell_cmd},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_shell_go_no_tests_to_run_does_not_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {
            "cmd": cmd,
            "effective_cmd": cmd,
            "exit_code": 0,
            "stdout": "ok\texample/pkg\t0.002s [no tests to run]\n",
            "stderr": "",
        }

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["go test ./..."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-shell-go-no-tests",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "main.go", "content": "package main\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": _VERIFY_GO_NO_TESTS_COMMAND},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-shell-go-no-tests.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


def test_one_shot_verify_run_go_mixed_output_counts(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["go test ./..."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-verify-run-go-mixed-output",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "main.go", "content": "package main\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_GO_MIXED_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_shell_go_mixed_output_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {
            "cmd": cmd,
            "effective_cmd": cmd,
            "exit_code": 0,
            "stdout": "?   \texample/pkg1\t[no test files]\nok  \texample/pkg2\t0.002s\n",
            "stderr": "",
        }

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["go test ./..."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-shell-go-mixed-output",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "main.go", "content": "package main\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": "go test ./..."},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 0


def test_one_shot_shell_go_no_test_files_does_not_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shell_run(
        *, root: Path, cmd: str, cwd: str | None = None, runner=None
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {
            "cmd": cmd,
            "effective_cmd": cmd,
            "exit_code": 0,
            "stdout": "?   \texample/pkg\t[no test files]\n",
            "stderr": "",
        }

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["go test ./..."],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-shell-go-no-test-files",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "main.go", "content": "package main\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={"cmd": _VERIFY_GO_NO_TEST_FILES_COMMAND},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change and ran verification.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-shell-go-no-test-files.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    problems = set(payload.get("problems") or [])
    assert "verification_failed" in problems


def test_one_shot_completion_gate_step_budget_exhausted_emits_forced_final_summary(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="code_only",
        verify_commands=["pytest -q"],
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-completion-gate-step-budget",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "main.py", "content": "print('done')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the requested code change.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content=(
                    "Completed work: updated `main.py`.\n"
                    "Remaining work: verification still needs to run.\n"
                    "Known issues or risks: the turn hit the step budget before completion."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-completion-gate-step-budget.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_completion_gate_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("reason") == "completion_gate_incomplete_verification_step_budget_exhausted"
    assert not any(
        event.get("type") == "error"
        and str((event.get("payload") or {}).get("error")) == "max_steps exceeded"
        for event in events
    )
    summary = _assert_forced_final_summary_emitted(events)
    assert "Remaining work: verification still needs to run." in summary


def test_one_shot_session_context_includes_follow_through_guidance(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
        one_shot_execution=True,
    )
    try:
        system_prompt = str(session.messages[0].get("content") or "")
        assert "One-shot execution mode" in system_prompt

        env_message = next(
            (
                str(message.get("content") or "")
                for message in session.messages
                if message.get("role") == "user"
                and "<environment_context>" in str(message.get("content") or "")
            ),
            "",
        )
        assert "one_shot_execution: true" in env_message
        assert "one_shot_guidance:" in env_message
    finally:
        session.close()


def test_one_shot_repo_turn_includes_pinned_task_brief(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo_with_commit(repo)
    _write_repo_text(repo, "src/app.py", "def main() -> None:\n    pass\n")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=repo,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
        one_shot_execution=True,
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="The module currently keeps the public API unchanged.",
                tool_calls=[],
                raw={},
            )
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        startup_task_brief = next(
            (
                str(message.get("content") or "")
                for message in session.messages
                if str(message.get("content") or "").startswith("<task_brief>")
            ),
            "",
        )
        exit_code = session.run_turn(
            "Explain how src/app.py works without changing the public API."
        )
    finally:
        session.close()

    task_brief = next(
        (
            str(message.get("content") or "")
            for message in client.call_records[0]["messages"]
            if str(message.get("role") or "") == "user"
            and str(message.get("content") or "").startswith("<task_brief>")
        ),
        "",
    )

    assert exit_code == 0
    assert "status: awaiting_substantive_repo_request" in startup_task_brief
    assert "Explain how src/app.py works without changing the public API." in task_brief


def test_one_shot_repo_session_preserves_anchored_focus_over_unanchored_follow_up(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo_with_commit(repo)
    _write_repo_text(repo, "src/parser.py", "def parse() -> str:\n    return 'ok'\n")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=repo,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
        one_shot_execution=True,
    )
    try:
        session.messages.append(
            {
                "role": "user",
                "content": "Fix src/parser.py without changing the CSV shape.",
            }
        )
        refreshed = agent_loop_mod.refresh_session_task_brief_message(
            session,
            pending_instruction="Also preserve unknown values like pending.",
        )
    finally:
        session.close()

    task_brief = next(
        (
            str(message.get("content") or "")
            for message in session.messages
            if str(message.get("content") or "").startswith("<task_brief>")
        ),
        "",
    )

    assert refreshed is True
    assert "current_focus:" in task_brief
    assert "- Fix src/parser.py without changing the CSV shape." in task_brief
    assert "recent_user_constraints:" in task_brief
    assert "- Also preserve unknown values like pending." in task_brief


def test_one_shot_repo_session_ignores_generic_long_follow_up(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo_with_commit(repo)
    _write_repo_text(repo, "src/parser.py", "def parse() -> str:\n    return 'ok'\n")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=repo,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
        one_shot_execution=True,
    )
    try:
        agent_loop_mod.refresh_session_task_brief_message(
            session,
            pending_instruction="Fix src/parser.py without changing the CSV shape.",
        )
        session.messages.append(
            {
                "role": "user",
                "content": "Fix src/parser.py without changing the CSV shape.",
            }
        )
        refreshed = agent_loop_mod.refresh_session_task_brief_message(
            session,
            pending_instruction="please keep going and explain a bit more first",
        )
    finally:
        session.close()

    task_brief = next(
        (
            str(message.get("content") or "")
            for message in session.messages
            if str(message.get("content") or "").startswith("<task_brief>")
        ),
        "",
    )

    assert refreshed is False
    assert "current_focus:" in task_brief
    assert "- Fix src/parser.py without changing the CSV shape." in task_brief
    assert "please keep going and explain a bit more first" not in task_brief


def test_one_shot_repo_session_keeps_concise_real_constraint(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo_with_commit(repo)
    _write_repo_text(repo, "src/parser.py", "def parse() -> str:\n    return 'ok'\n")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=repo,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
        one_shot_execution=True,
    )
    try:
        agent_loop_mod.refresh_session_task_brief_message(
            session,
            pending_instruction="Fix src/parser.py without changing the CSV shape.",
        )
        session.messages.append(
            {
                "role": "user",
                "content": "Fix src/parser.py without changing the CSV shape.",
            }
        )
        refreshed = agent_loop_mod.refresh_session_task_brief_message(
            session,
            pending_instruction="Handle empty lines too.",
        )
    finally:
        session.close()

    task_brief = next(
        (
            str(message.get("content") or "")
            for message in session.messages
            if str(message.get("content") or "").startswith("<task_brief>")
        ),
        "",
    )

    assert refreshed is True
    assert "- Fix src/parser.py without changing the CSV shape." in task_brief
    assert "- Handle empty lines too." in task_brief


def test_one_shot_repo_session_promotes_strong_unanchored_task_shift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo_with_commit(repo)
    _write_repo_text(repo, "src/parser.py", "def parse() -> str:\n    return 'ok'\n")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=repo,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
        one_shot_execution=True,
    )
    try:
        session.messages.append(
            {
                "role": "user",
                "content": "Fix src/parser.py without changing the CSV shape.",
            }
        )
        refreshed = agent_loop_mod.refresh_session_task_brief_message(
            session,
            pending_instruction="Actually add a regression test and keep API stable.",
        )
    finally:
        session.close()

    task_brief = next(
        (
            str(message.get("content") or "")
            for message in session.messages
            if str(message.get("content") or "").startswith("<task_brief>")
        ),
        "",
    )

    assert refreshed is True
    assert "current_focus:" in task_brief
    assert "- Actually add a regression test and keep API stable." in task_brief
    assert "recent_user_constraints:" in task_brief
    assert "- Fix src/parser.py without changing the CSV shape." in task_brief


def test_one_shot_repo_task_brief_keeps_existing_focus_for_anchored_explanatory_follow_up(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo_with_commit(repo)
    _write_repo_text(repo, "src/parser.py", "def parse() -> str:\n    return 'ok'\n")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=repo,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
        one_shot_execution=True,
    )
    try:
        session.messages.append(
            {
                "role": "user",
                "content": (
                    "Fix src/parser.py without changing the CSV shape. Keep the public API stable."
                ),
            }
        )
        refreshed = agent_loop_mod.refresh_session_task_brief_message(
            session,
            pending_instruction="Can you explain more about src/parser.py?",
        )
    finally:
        session.close()

    task_brief = next(
        (
            str(message.get("content") or "")
            for message in session.messages
            if str(message.get("content") or "").startswith("<task_brief>")
        ),
        "",
    )

    assert refreshed is True
    assert "current_focus:" in task_brief
    assert (
        "- Fix src/parser.py without changing the CSV shape. Keep the public API stable."
        in task_brief
    )
    assert "- Can you explain more about src/parser.py?" in task_brief


def _write_test_files(root: Path, names: list[str]) -> None:
    for name in names:
        file_path = root / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"sample content for {name}\n", encoding="utf-8")


def _install_stub_subagent_run(
    session: Any,
    *,
    result_text: str,
) -> None:
    original = session.tools["subagent_run"]

    def _run(args: dict[str, Any]) -> dict[str, Any]:
        subagent_name = str(args.get("name") or "explorer").strip() or "explorer"
        return {
            "subagent": subagent_name,
            "subagent_session_id": "stub-subagent-session",
            "result": result_text,
            "usage": {"total_tokens": 10},
            "sandbox": {"mode": "readonly", "tools": ["fs_read", "search_rg"]},
        }

    session.tools["subagent_run"] = ToolDef(
        name=original.name,
        description=original.description,
        parameters=original.parameters,
        run=_run,
    )


def test_one_shot_exploration_stagnation_emits_nudge(tmp_path: Path) -> None:
    _write_test_files(tmp_path, [f"f{i}.txt" for i in range(1, 7)])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=14,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-exploration-stagnation",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_read", arguments={"path": "f1.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="fs_read", arguments={"path": "f2.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="fs_read", arguments={"path": "f3.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc4", name="fs_read", arguments={"path": "f4.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc5", name="fs_read", arguments={"path": "f5.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc6", name="fs_read", arguments={"path": "f6.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc7", name="fs_write", arguments={"path": "out.txt", "content": "done"}
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc8",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-exploration-stagnation.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_exploration_stagnation_detected" in event_types
    assert "exploration_nudge" in event_types
    assert "one_shot_exploration_incomplete_after_retries" not in event_types


def test_one_shot_post_explore_stagnation_emits_implementation_bootstrap_nudge(
    tmp_path: Path,
) -> None:
    _write_test_files(tmp_path, ["src/mini_notes/cli.py"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        subagents_enabled=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-post-explore-bootstrap-nudge",
    )
    _install_stub_subagent_run(
        session,
        result_text=(
            "Most likely edit targets now: src/mini_notes/cli.py, src/mini_notes/logic.py, "
            "tests/test_cli.py, README.md."
        ),
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="subagent_run",
                        arguments={"name": "general-purpose", "task": "Map repo"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc2", name="fs_read", arguments={"path": "src/mini_notes/cli.py"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc3", name="fs_read", arguments={"path": "src/mini_notes/cli.py"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc4", name="fs_read", arguments={"path": "src/mini_notes/cli.py"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc5",
                        name="fs_write",
                        arguments={"path": "out.txt", "content": "implemented"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc6",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-post-explore-bootstrap-nudge.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_post_explore_stagnation_detected" in event_types
    bootstrap_events = [
        event for event in events if event.get("type") == "implementation_bootstrap_nudge"
    ]
    assert bootstrap_events
    assert not any(event.get("type") == "no_material_edits_bootstrap_nudge" for event in events)
    payload = dict(bootstrap_events[-1].get("payload") or {})
    message = str(payload.get("message") or "")
    assert "Do not call the same research subagent again in this turn." in message
    assert "subagent_run(name=explorer)" not in message


def test_one_shot_post_explore_bootstrap_recovers_after_real_action(tmp_path: Path) -> None:
    _write_test_files(tmp_path, ["src/mini_notes/logic.py", "src/mini_notes/cli.py"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=16,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        subagents_enabled=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-post-explore-recovery",
    )
    _install_stub_subagent_run(
        session,
        result_text="Edit targets: src/mini_notes/logic.py, src/mini_notes/cli.py, tests/test_cli.py",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="subagent_run",
                        arguments={"name": "explorer", "task": "Map repo"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc5",
                        name="fs_write",
                        arguments={"path": "src/mini_notes/logic.py", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc6",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-post-explore-recovery.jsonl"))
    assert any(event.get("type") == "implementation_bootstrap_nudge" for event in events)
    assert not any(
        event.get("type") == "one_shot_post_explore_incomplete_after_retries" for event in events
    )


def test_one_shot_greek_post_explore_progress_transitions_to_action(tmp_path: Path) -> None:
    _write_test_files(tmp_path, ["src/mini_notes/logic.py", "src/mini_notes/cli.py"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=14,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        subagents_enabled=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-greek-post-explore-flow",
    )
    _install_stub_subagent_run(
        session,
        result_text=(
            "Edit targets: src/mini_notes/logic.py, src/mini_notes/cli.py, "
            "tests/test_cli.py, README.md"
        ),
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="subagent_run",
                        arguments={"name": "explorer", "task": "Map repo"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content=(
                    "Θα προχωρήσω στην υλοποίηση. Επόμενο βήμα: "
                    "θα αλλάξω το src/mini_notes/logic.py."
                ),
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_write",
                        arguments={"path": "src/mini_notes/logic.py", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Υλοποίησα την αλλαγή, ενημέρωσα το README και έτρεξα τα tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Υλοποίησε αυτή την αλλαγή και ενημέρωσε docs/tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-greek-post-explore-flow.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_non_final_progress_detected" in event_types
    assert "continuation_nudge" in event_types
    assert "one_shot_incomplete_after_retries" not in event_types


def test_one_shot_post_explore_retry_exhausted_emits_forced_final_summary(
    tmp_path: Path,
) -> None:
    _write_test_files(tmp_path, ["src/mini_notes/cli.py"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        subagents_enabled=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-post-explore-fail-cap",
        surface=surface,
    )
    _install_stub_subagent_run(
        session,
        result_text="Edit targets: src/mini_notes/cli.py, src/mini_notes/logic.py, tests/test_cli.py",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="subagent_run",
                        arguments={"name": "explorer", "task": "Map repo"},
                    )
                ],
                raw={},
            ),
            *[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"tc{idx}",
                            name="fs_read",
                            arguments={"path": "src/mini_notes/cli.py"},
                        )
                    ],
                    raw={},
                )
                for idx in range(2, 9)
            ],
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-post-explore-fail-cap.jsonl"))
    assert any(event.get("type") == "one_shot_post_explore_stagnation_detected" for event in events)
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_post_explore_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("reason") == "repeated_successful_exploration_loop"
    assert not any(
        event.get("type") == "error"
        and str((event.get("payload") or {}).get("error")) == "max_steps exceeded"
        for event in events
    )
    requested = [event for event in events if event.get("type") == "forced_final_summary_requested"]
    assert requested
    assert dict(requested[-1].get("payload") or {}).get("reason") == "post_explore_retry_exhausted"
    expected_error = (
        "One-shot run stopped: post-explore stagnation persisted after bounded "
        "implementation-bootstrap nudges. Start implementing now or report a concrete blocker."
    )
    assert expected_error in surface.errors
    assert surface.errors[-1] == expected_error
    summary = _assert_forced_final_summary_emitted(events)
    assert surface.final_messages[-1] == summary


def test_one_shot_post_explore_bootstrap_nudge_includes_recent_path_anchors(
    tmp_path: Path,
) -> None:
    _write_test_files(
        tmp_path,
        [
            "src/mini_notes/logic.py",
            "src/mini_notes/cli.py",
            "tests/test_logic.py",
            "README.md",
        ],
    )

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        subagents_enabled=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-post-explore-path-anchors",
    )
    _install_stub_subagent_run(
        session,
        result_text=(
            "Targets now: src/mini_notes/logic.py, src/mini_notes/cli.py, tests/test_logic.py, README.md."
        ),
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="subagent_run",
                        arguments={"name": "explorer", "task": "Map repo"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="fs_read",
                        arguments={"path": "src/mini_notes/logic.py"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc5",
                        name="fs_write",
                        arguments={"path": "src/mini_notes/logic.py", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc6",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-post-explore-path-anchors.jsonl"))
    bootstrap_events = [
        event for event in events if event.get("type") == "implementation_bootstrap_nudge"
    ]
    assert bootstrap_events
    payload = dict(bootstrap_events[-1].get("payload") or {})
    anchor_paths = list(payload.get("anchor_paths") or [])
    assert "src/mini_notes/logic.py" in anchor_paths
    assert "src/mini_notes/cli.py" in anchor_paths


def test_one_shot_repeated_successful_identical_read_loop_triggers_guard(tmp_path: Path) -> None:
    _write_test_files(tmp_path, ["repeat.txt"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=10,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-repeated-read-loop",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_read", arguments={"path": "repeat.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="fs_read", arguments={"path": "repeat.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="fs_read", arguments={"path": "repeat.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4", name="fs_write", arguments={"path": "out.txt", "content": "done"}
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc5",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-repeated-read-loop.jsonl"))
    detections = [
        event for event in events if event.get("type") == "one_shot_exploration_stagnation_detected"
    ]
    assert detections
    payload = dict(detections[-1].get("payload") or {})
    assert payload.get("reason") == "repeated_successful_exploration_loop"


def test_one_shot_action_progress_resets_exploration_stagnation_counters(tmp_path: Path) -> None:
    _write_test_files(tmp_path, [f"f{i}.txt" for i in range(1, 10)])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=20,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-reset-counters",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_read", arguments={"path": "f1.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="fs_read", arguments={"path": "f2.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="fs_read", arguments={"path": "f3.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc4", name="fs_read", arguments={"path": "f4.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc5", name="fs_read", arguments={"path": "f5.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc6", name="fs_read", arguments={"path": "f6.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc7", name="fs_write", arguments={"path": "out.txt", "content": "done"}
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc8", name="fs_read", arguments={"path": "f7.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc9", name="fs_read", arguments={"path": "f8.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc10", name="fs_read", arguments={"path": "f9.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc11",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-reset-counters.jsonl"))
    nudge_events = [event for event in events if event.get("type") == "exploration_nudge"]
    assert len(nudge_events) == 1
    assert all(
        event.get("type") != "one_shot_exploration_incomplete_after_retries" for event in events
    )


def test_one_shot_plan_only_task_does_not_trigger_exploration_stagnation_nudges(
    tmp_path: Path,
) -> None:
    _write_test_files(tmp_path, ["f1.txt", "f2.txt"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-plan-only-no-exploration-nudge",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_read", arguments={"path": "f1.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="fs_read", arguments={"path": "f2.txt"})],
                raw={},
            ),
            LLMResponse(
                content="Plan: 1) inspect files 2) update parser 3) test", tool_calls=[], raw={}
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Give me a plan only; do not modify files.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(
        read_session_events(sessions_dir / "one-shot-plan-only-no-exploration-nudge.jsonl")
    )
    event_types = [event.get("type") for event in events]
    assert "one_shot_exploration_stagnation_detected" not in event_types
    assert "exploration_nudge" not in event_types


def test_one_shot_exploration_retry_exhausted_emits_forced_final_summary(tmp_path: Path) -> None:
    _write_test_files(tmp_path, ["repeat.txt"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=10,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-exploration-fail-after-nudges",
        surface=surface,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_read", arguments={"path": "repeat.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="fs_read", arguments={"path": "repeat.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc3", name="fs_read", arguments={"path": "repeat.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc4", name="fs_read", arguments={"path": "repeat.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc5", name="fs_read", arguments={"path": "repeat.txt"})],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(
        read_session_events(sessions_dir / "one-shot-exploration-fail-after-nudges.jsonl")
    )
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_exploration_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("reason") == "repeated_successful_exploration_loop"
    requested = [event for event in events if event.get("type") == "forced_final_summary_requested"]
    assert requested
    assert dict(requested[-1].get("payload") or {}).get("reason") == "exploration_retry_exhausted"
    expected_error = (
        "One-shot run stopped: exploration stagnation persisted after bounded nudges. "
        "Start implementing, delegate once to a suitable available subagent if more "
        "investigation is genuinely needed, or report a concrete blocker."
    )
    assert expected_error in surface.errors
    assert surface.errors[-1] == expected_error
    summary = _assert_forced_final_summary_emitted(events)
    assert surface.final_messages[-1] == summary


def test_one_shot_failed_varied_read_loop_step_budget_exhausted(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-failed-varied-read-loop",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="fs_read", arguments={"path": "missing-1.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc2", name="fs_read", arguments={"path": "missing-2.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc3", name="fs_read", arguments={"path": "missing-3.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc4", name="fs_read", arguments={"path": "missing-4.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc5", name="fs_read", arguments={"path": "missing-5.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc6", name="fs_read", arguments={"path": "missing-6.txt"})
                ],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-failed-varied-read-loop.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_exploration_stagnation_detected" in event_types
    assert "exploration_nudge" in event_types
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_exploration_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("reason") == "exploration_step_budget_exhausted"
    assert payload.get("exploration_attempt_outcome") == "failed"
    assert not any(
        event.get("type") == "error"
        and str((event.get("payload") or {}).get("error")) == "max_steps exceeded"
        for event in events
    )
    _assert_forced_final_summary_emitted(events)


def test_one_shot_failed_exploration_then_action_resets_and_recovers(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=14,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-failed-recovery",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="fs_read", arguments={"path": "missing-a1.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc2", name="fs_read", arguments={"path": "missing-a2.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc3", name="fs_read", arguments={"path": "missing-a3.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc4", name="fs_read", arguments={"path": "missing-a4.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc5", name="fs_read", arguments={"path": "missing-a5.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc6",
                        name="fs_write",
                        arguments={"path": "out.txt", "content": "implemented"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc7", name="fs_read", arguments={"path": "missing-b1.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc8", name="fs_read", arguments={"path": "missing-b2.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc9", name="fs_read", arguments={"path": "missing-b3.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc10", name="fs_read", arguments={"path": "missing-b4.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc11", name="fs_read", arguments={"path": "missing-b5.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc12",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.", tool_calls=[], raw={}
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-failed-recovery.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "exploration_nudge" not in event_types
    assert "one_shot_exploration_incomplete_after_retries" not in event_types


def test_one_shot_mixed_success_and_failed_exploration_triggers_guard(tmp_path: Path) -> None:
    _write_test_files(tmp_path, ["existing.txt"])

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-mixed-exploration-loop",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1a", name="fs_read", arguments={"path": "existing.txt"}),
                    ToolCall(id="tc1b", name="fs_read", arguments={"path": "missing-1.txt"}),
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc2a", name="fs_read", arguments={"path": "existing.txt"}),
                    ToolCall(id="tc2b", name="fs_read", arguments={"path": "missing-2.txt"}),
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc3a", name="fs_read", arguments={"path": "existing.txt"}),
                    ToolCall(id="tc3b", name="fs_read", arguments={"path": "missing-3.txt"}),
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc4a", name="fs_read", arguments={"path": "existing.txt"}),
                    ToolCall(id="tc4b", name="fs_read", arguments={"path": "missing-4.txt"}),
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc5a", name="fs_read", arguments={"path": "existing.txt"}),
                    ToolCall(id="tc5b", name="fs_read", arguments={"path": "missing-5.txt"}),
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc6a", name="fs_read", arguments={"path": "existing.txt"}),
                    ToolCall(id="tc6b", name="fs_read", arguments={"path": "missing-6.txt"}),
                ],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-mixed-exploration-loop.jsonl"))
    detections = [
        event for event in events if event.get("type") == "one_shot_exploration_stagnation_detected"
    ]
    assert detections
    payload = dict(detections[-1].get("payload") or {})
    assert payload.get("exploration_attempt_outcome") == "mixed"
    assert payload.get("step_exploration_attempt_outcome") == "mixed"
    assert any(event.get("type") == "exploration_nudge" for event in events)


def test_one_shot_plan_only_failed_exploration_does_not_trigger_nudges(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-plan-only-failed-exploration",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="fs_read", arguments={"path": "missing-1.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc2", name="fs_read", arguments={"path": "missing-2.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc3", name="fs_read", arguments={"path": "missing-3.txt"})
                ],
                raw={},
            ),
            LLMResponse(
                content="Plan: inspect key modules and identify extension points.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Give me a plan only; do not modify files.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-plan-only-failed-exploration.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_exploration_stagnation_detected" not in event_types
    assert "exploration_nudge" not in event_types
    assert "one_shot_exploration_incomplete_after_retries" not in event_types


def test_one_shot_failed_exploration_fails_clearly_after_nudge_cap(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=10,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-failed-exploration-cap",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"tc{i}",
                        name="fs_read",
                        arguments={"path": "missing-repeat.txt"},
                    )
                ],
                raw={},
            )
            for i in range(1, 13)
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-failed-exploration-cap.jsonl"))
    incomplete_events = [
        event
        for event in events
        if event.get("type") == "one_shot_exploration_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("reason") == "repeated_failed_exploration_loop"
    assert payload.get("exploration_attempt_outcome") == "failed"
    assert not any(
        event.get("type") == "error"
        and str((event.get("payload") or {}).get("error")) == "max_steps exceeded"
        for event in events
    )


def test_one_shot_failed_edit_stagnation_emits_strategy_nudge_and_recovers(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=10,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-edit-recovery",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_read", arguments={"path": "target.txt"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_edit",
                        arguments={
                            "path": "target.txt",
                            "edits": [
                                {
                                    "op": "replace_exact",
                                    "target": "beta",
                                    "replacement": "gamma",
                                }
                            ],
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_edit",
                        arguments={
                            "path": "target.txt",
                            "edits": [
                                {
                                    "op": "replace_wrong",
                                    "target": "alpha",
                                    "replacement": "ALPHA",
                                }
                            ],
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="fs_edit",
                        arguments={
                            "path": "target.txt",
                            "edits": [
                                {
                                    "op": "replace_wrong",
                                    "target": "alpha",
                                    "replacement": "ALPHA",
                                }
                            ],
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc5",
                        name="fs_write",
                        arguments={"path": "target.txt", "content": "ALPHA\ngamma\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc6",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    assert target.read_text(encoding="utf-8") == "ALPHA\ngamma\n"
    events = list(read_session_events(sessions_dir / "one-shot-edit-recovery.jsonl"))
    event_types = [event.get("type") for event in events]
    assert "one_shot_edit_stagnation_detected" in event_types
    assert "edit_strategy_nudge" in event_types
    assert "one_shot_edit_incomplete_after_retries" not in event_types


def test_one_shot_edit_retry_exhausted_emits_forced_final_summary(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    surface = _RecordingSurface()
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-edit-fail-cap",
        surface=surface,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"tc{i}",
                        name="fs_edit",
                        arguments={
                            "path": "target.txt",
                            "edits": [
                                {
                                    "op": "replace_wrong",
                                    "target": "alpha",
                                    "replacement": "ALPHA",
                                }
                            ],
                        },
                    )
                ],
                raw={},
            )
            for i in range(1, 13)
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 1
    events = list(read_session_events(sessions_dir / "one-shot-edit-fail-cap.jsonl"))
    incomplete_events = [
        event for event in events if event.get("type") == "one_shot_edit_incomplete_after_retries"
    ]
    assert incomplete_events
    payload = dict(incomplete_events[-1].get("payload") or {})
    assert payload.get("reason") == "repeated_failed_edit_loop"
    assert not any(
        event.get("type") == "error"
        and str((event.get("payload") or {}).get("error")) == "max_steps exceeded"
        for event in events
    )
    requested = [event for event in events if event.get("type") == "forced_final_summary_requested"]
    assert requested
    assert dict(requested[-1].get("payload") or {}).get("reason") == "edit_retry_exhausted"
    expected_error = (
        "One-shot run stopped: failed edit/write loop persisted after bounded strategy "
        "nudges. Switch to exact-match fs_edit ops, or use git_apply_patch/fs_write, "
        "or report a concrete blocker."
    )
    assert expected_error in surface.errors
    assert surface.errors[-1] == expected_error
    summary = _assert_forced_final_summary_emitted(events)
    assert surface.final_messages[-1] == summary


def test_one_shot_successful_write_resets_failed_edit_stagnation_counters(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    cfg = AppConfig(model="test-model", routing_mode="code_only")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=14,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="one-shot-edit-reset-counters",
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_edit",
                        arguments={
                            "path": "target.txt",
                            "edits": [
                                {
                                    "op": "replace_wrong",
                                    "target": "alpha",
                                    "replacement": "ALPHA",
                                }
                            ],
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_edit",
                        arguments={
                            "path": "target.txt",
                            "edits": [
                                {
                                    "op": "replace_wrong",
                                    "target": "alpha",
                                    "replacement": "ALPHA",
                                }
                            ],
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="fs_write",
                        arguments={"path": "target.txt", "content": "ALPHA\ngamma\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc4",
                        name="fs_edit",
                        arguments={
                            "path": "target.txt",
                            "edits": [
                                {
                                    "op": "replace_wrong",
                                    "target": "ALPHA",
                                    "replacement": "alpha",
                                }
                            ],
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc5",
                        name="fs_edit",
                        arguments={
                            "path": "target.txt",
                            "edits": [
                                {
                                    "op": "replace_wrong",
                                    "target": "ALPHA",
                                    "replacement": "alpha",
                                }
                            ],
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc6",
                        name="fs_write",
                        arguments={"path": "target.txt", "content": "done\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc7",
                        name="verify_run",
                        arguments={"commands": [_VERIFY_OK_COMMAND]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented search, updated README, and ran tests.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
    finally:
        session.close()

    assert exit_code == 0
    events = list(read_session_events(sessions_dir / "one-shot-edit-reset-counters.jsonl"))
    nudge_events = [event for event in events if event.get("type") == "edit_strategy_nudge"]
    assert len(nudge_events) == 2
    assert [int((event.get("payload") or {}).get("attempt") or 0) for event in nudge_events] == [
        1,
        1,
    ]
    assert all(event.get("type") != "one_shot_edit_incomplete_after_retries" for event in events)
