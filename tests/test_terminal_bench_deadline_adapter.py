from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.terminal_bench import sylliptor_agent as adapter_mod
from benchmarks.terminal_bench.harbor_agent import SylliptorAgent
from benchmarks.terminal_bench.sylliptor_agent import (
    MANAGED_HOST_DEADLINE_DIAGNOSTIC_FILENAME,
    MANAGED_HOST_SHUTDOWN_RESERVE_ENV,
    VERIFY_CMD_ENV,
    SylliptorSimpleAgent,
)
from sylliptor_agent_cli.run_outcome import INFRASTRUCTURE_FAILURE_EXIT_CODE, RunOutcome


class _SessionSpy:
    def __init__(self) -> None:
        self.copied: list[tuple[Any, dict[str, Any]]] = []
        self.commands: list[Any] = []

    def copy_to_container(self, *args: Any, **kwargs: Any) -> None:
        self.copied.append((args, kwargs))

    def send_command(self, command: Any) -> None:
        self.commands.append(command)


def _agent(**kwargs: Any) -> SylliptorSimpleAgent:
    defaults: dict[str, Any] = {
        "api_key": "SECRET-KEY",
        "model_name": "test-model",
        "base_url": "https://example.invalid/v1",
        "managed_host_agent_timeout_sec": 100,
        "managed_host_shutdown_reserve_sec": 10,
    }
    defaults.update(kwargs)
    return SylliptorSimpleAgent(**defaults)


def _split_command(agent: SylliptorSimpleAgent, instruction: str) -> tuple[list[str], Any]:
    command = agent._run_agent_commands(instruction)[0]
    return shlex.split(command.command, posix=True), command


def test_adapter_command_includes_required_deadline_flags_and_separator() -> None:
    instruction = "--starts-with-dash\nquote 'x' and shell $(echo nope) unicode: Δοκιμή"
    parts, command = _split_command(_agent(), instruction)

    assert parts[:2] == ["sylliptor", "run"]
    assert parts.count("--deadline-seconds") == 1
    deadline_index = parts.index("--deadline-seconds")
    assert parts[deadline_index + 1] == "90"
    assert parts.count("--require-deadline") == 1
    separator_index = parts.index("--")
    assert parts[separator_index + 1 :] == [instruction]
    assert parts.index("--require-deadline") < separator_index
    assert command.max_timeout_sec == 101
    assert "SECRET-KEY" not in command.command


def test_adapter_default_has_no_host_verify_command() -> None:
    agent = _agent()
    parts, _command = _split_command(agent, "do work")

    assert "--verify-cmd" not in parts
    assert VERIFY_CMD_ENV not in agent._env


def test_adapter_appends_one_verify_flag_per_explicit_command() -> None:
    agent = _agent(verify_cmd=["pytest -q", "ruff check ."])
    parts, _command = _split_command(agent, "do work")

    verify_indices = [index for index, part in enumerate(parts) if part == "--verify-cmd"]
    assert [parts[index + 1] for index in verify_indices] == ["pytest -q", "ruff check ."]
    assert VERIFY_CMD_ENV not in agent._env


def test_adapter_single_explicit_verify_command_is_exported_for_setup() -> None:
    agent = _agent(verify_cmd="pytest -q")
    parts, _command = _split_command(agent, "do work")

    assert parts[parts.index("--verify-cmd") + 1] == "pytest -q"
    assert agent._env[VERIFY_CMD_ENV] == "pytest -q"


def test_adapter_rejects_simultaneous_verify_cmd_and_verify_cmds() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _agent(verify_cmd="pytest -q", verify_cmds=["ruff check ."])


def test_adapter_rejects_empty_explicit_verify_command() -> None:
    with pytest.raises(ValueError, match="verifier command is empty"):
        _agent(verify_cmd="")


def test_adapter_rejects_empty_member_in_explicit_verify_commands() -> None:
    with pytest.raises(ValueError, match="verifier command is empty"):
        _agent(verify_cmd=["pytest -q", " "])


def test_adapter_rejects_unordered_verify_command_set() -> None:
    with pytest.raises(ValueError, match="ordered sequence"):
        _agent(verify_cmd={"pytest -q", "ruff check ."})


def test_adapter_rejects_vacuous_explicit_host_verifier_before_setup(tmp_path: Path) -> None:
    agent = _agent(verify_cmd="true")
    session = _SessionSpy()

    with pytest.raises(ValueError, match="vacuous_verifier"):
        agent.perform_task("do not launch", session, logging_dir=tmp_path)

    assert session.copied == []
    assert session.commands == []
    record = json.loads(
        (tmp_path / MANAGED_HOST_DEADLINE_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert record["status"] == "blocked"
    assert record["validation_error"] == "invalid_host_verifier"
    assert record["host_verifier_status"] == "provided"
    assert record["host_verifier_count"] == 1
    assert record["host_verifier_rejection_reasons"] == ["vacuous_verifier"]
    serialized = json.dumps(record)
    assert "do not launch" not in serialized
    assert "SECRET-KEY" not in serialized
    assert "true" not in serialized


def test_adapter_uses_monotonic_elapsed_time_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent = _agent(managed_host_agent_timeout_sec=100, managed_host_shutdown_reserve_sec=10)
    agent._managed_host_started_at_monotonic = 50.0
    agent._managed_host_logging_dir = tmp_path
    monkeypatch.setattr(adapter_mod.time, "monotonic", lambda: 62.5)

    parts, command = _split_command(agent, "do work")

    assert parts[parts.index("--deadline-seconds") + 1] == "77.5"
    assert command.max_timeout_sec == 88.5
    record = json.loads(
        (tmp_path / MANAGED_HOST_DEADLINE_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert record["status"] == "ok"
    assert record["final_effective_host_agent_timeout_seconds"] == 100.0
    assert record["elapsed_before_launch_seconds"] == 12.5
    assert record["host_shutdown_reserve_seconds"] == 10.0
    assert record["sylliptor_invocation_deadline_seconds"] == 77.5
    assert record["terminal_command_timeout_seconds"] == 88.5
    assert record["host_verifier_status"] == "unavailable"
    assert record["host_verifier_count"] == 0


def test_adapter_shutdown_reserve_kwarg_takes_precedence_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MANAGED_HOST_SHUTDOWN_RESERVE_ENV, "7")
    parts, _command = _split_command(
        _agent(managed_host_agent_timeout_sec=100, managed_host_shutdown_reserve_sec=12),
        "do work",
    )

    assert parts[parts.index("--deadline-seconds") + 1] == "88"


def test_adapter_uses_shutdown_reserve_environment_when_kwarg_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MANAGED_HOST_SHUTDOWN_RESERVE_ENV, "7.5")
    agent = SylliptorSimpleAgent(
        api_key="SECRET-KEY",
        model_name="test-model",
        base_url="https://example.invalid/v1",
        managed_host_agent_timeout_sec=100,
    )

    parts, _command = _split_command(agent, "do work")

    assert parts[parts.index("--deadline-seconds") + 1] == "92.5"


def test_adapter_fail_closed_when_authoritative_host_timeout_missing(
    tmp_path: Path,
) -> None:
    agent = SylliptorSimpleAgent(
        api_key="SECRET-KEY",
        model_name="test-model",
        base_url="https://example.invalid/v1",
    )
    session = _SessionSpy()

    with pytest.raises(ValueError, match="managed-host deadline"):
        agent.perform_task("do not launch", session, logging_dir=tmp_path)

    assert session.copied == []
    assert session.commands == []
    record = json.loads(
        (tmp_path / MANAGED_HOST_DEADLINE_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert record["status"] == "blocked"
    assert record["timeout_source"] == "absent"
    assert record["validation_error"] == "final_effective_host_agent_timeout_seconds_missing"
    serialized = json.dumps(record)
    assert "do not launch" not in serialized
    assert "SECRET-KEY" not in serialized


def test_adapter_fail_closed_when_elapsed_time_consumes_launch_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent = _agent(managed_host_agent_timeout_sec=20, managed_host_shutdown_reserve_sec=5)
    agent._managed_host_started_at_monotonic = 10.0
    agent._managed_host_logging_dir = tmp_path
    monkeypatch.setattr(adapter_mod.time, "monotonic", lambda: 24.5)

    with pytest.raises(ValueError, match="too small to launch"):
        agent._run_agent_commands("do not launch")

    record = json.loads(
        (tmp_path / MANAGED_HOST_DEADLINE_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert record["status"] == "blocked"
    assert record["validation_error"] == "remaining_duration_too_small"
    assert record["sylliptor_invocation_deadline_seconds"] == 0.5


def test_adapter_outer_inner_boundary_keeps_host_reserve_available() -> None:
    parts, command = _split_command(
        _agent(managed_host_agent_timeout_sec=60, managed_host_shutdown_reserve_sec=8),
        "do work",
    )
    deadline_seconds = float(parts[parts.index("--deadline-seconds") + 1])
    simulated_host_remaining = command.max_timeout_sec - 1.0

    assert deadline_seconds == 52
    assert simulated_host_remaining == 60
    assert simulated_host_remaining - deadline_seconds == 8
    assert command.max_timeout_sec > deadline_seconds


def test_adapter_perform_task_launches_exactly_one_deadline_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_base_perform_task(
        self: SylliptorSimpleAgent,
        instruction: str,
        session: _SessionSpy,
        logging_dir: Path | None = None,
    ) -> Any:
        _ = logging_dir
        for command in self._run_agent_commands(instruction):
            session.send_command(command)
        return adapter_mod.AgentResult(total_input_tokens=0, total_output_tokens=0)

    monkeypatch.setattr(adapter_mod.AbstractInstalledAgent, "perform_task", fake_base_perform_task)
    agent = _agent(managed_host_agent_timeout_sec=100, managed_host_shutdown_reserve_sec=10)
    session = _SessionSpy()

    result = agent.perform_task("finish this", session, logging_dir=tmp_path)

    assert result.total_input_tokens == 0
    assert len(session.copied) == 1
    assert len(session.commands) == 1
    parts = shlex.split(session.commands[0].command, posix=True)
    assert parts.count("--deadline-seconds") == 1
    assert parts.count("--require-deadline") == 1


def test_legacy_adapter_requires_explicit_sylliptor_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "SYLLIPTOR_API_KEY",
        "SYLLIPTOR_BASE_URL",
        "SYLLIPTOR_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-inherited")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://must-not-be-inherited.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "must-not-be-inherited")

    agent = SylliptorSimpleAgent(
        managed_host_agent_timeout_sec=100,
        managed_host_shutdown_reserve_sec=10,
    )

    with pytest.raises(ValueError, match="Missing Terminal-Bench provider configuration") as exc:
        _ = agent._env

    message = str(exc.value)
    assert "SYLLIPTOR_API_KEY" in message
    assert "SYLLIPTOR_BASE_URL" in message
    assert "SYLLIPTOR_MODEL" in message
    assert "must-not-be-inherited" not in message

    monkeypatch.setenv("SYLLIPTOR_API_KEY", "ambient-key")
    monkeypatch.setenv("SYLLIPTOR_BASE_URL", "https://ambient.invalid/v1")
    monkeypatch.setenv("SYLLIPTOR_MODEL", "ambient-model")
    explicit_blanks = SylliptorSimpleAgent(
        api_key="",
        base_url="",
        model_name="",
        managed_host_agent_timeout_sec=100,
        managed_host_shutdown_reserve_sec=10,
    )
    with pytest.raises(ValueError, match="Missing Terminal-Bench provider configuration"):
        _ = explicit_blanks._env


def test_benchmark_setup_is_present_provider_neutral_and_public_safe() -> None:
    setup_path = _agent()._install_agent_script_path
    setup = setup_path.read_text(encoding="utf-8")

    assert setup_path.name == "setup.sh"
    assert "sylliptor-agent-cli" in setup
    assert "SYLLIPTOR_API_KEY" not in setup
    assert "SYLLIPTOR_BASE_URL" not in setup
    assert "SYLLIPTOR_MODEL" not in setup
    assert "retry apt-get-update apt-get update" in setup
    assert "retry apk-add apk add" in setup
    assert "retry dnf-install dnf install" in setup
    assert "retry uv-installer" in setup
    assert "retry pip-install-sylliptor" in setup
    assert "retry uv-pip-install" in setup
    assert "uv python install 3.12" in setup
    assert "python3 -m venv /opt/sylliptor-venv" in setup
    assert "--break-system-packages" not in setup


def test_public_benchmark_readme_covers_reproduction_and_secret_handling() -> None:
    readme = Path(adapter_mod.__file__).with_name("README.md").read_text(encoding="utf-8")

    assert "## Quick start with Harbor" in readme
    assert "## Reproducible reporting" in readme
    assert "## Security and privacy" in readme
    assert "SYLLIPTOR_API_KEY" in readme


def test_harbor_adapter_preserves_provider_qualified_model_identifier() -> None:
    agent = SylliptorAgent(
        model_name="provider/model-name",
        extra_env={"SYLLIPTOR_BASE_URL": "https://example.invalid/v1"},
    )

    assert agent._model() == "provider/model-name"


def test_harbor_adapter_version_identifies_local_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "sylliptor_agent_cli-1.2.3-py3-none-any.whl"
    wheel.touch()
    agent = SylliptorAgent(extra_env={"SYLLIPTOR_WHEEL": str(wheel)})

    assert agent.version() == wheel.stem


def test_harbor_adapter_explicit_benchmark_version_takes_precedence(tmp_path: Path) -> None:
    wheel = tmp_path / "sylliptor_agent_cli-1.2.3-py3-none-any.whl"
    wheel.touch()
    agent = SylliptorAgent(
        version="ignored-base-version",
        extra_env={
            "SYLLIPTOR_BENCH_VERSION": "commit-abc123",
            "SYLLIPTOR_WHEEL": str(wheel),
        },
    )

    assert agent.version() == "commit-abc123"


def test_harbor_adapter_extra_args_are_parsed_as_arguments() -> None:
    agent = SylliptorAgent(
        extra_env={"SYLLIPTOR_EXTRA_ARGS": "--temperature 0.3 --label 'two words'"}
    )

    assert agent._extra_cli_args() == ["--temperature", "0.3", "--label", "two words"]


def test_harbor_adapter_rejects_malformed_extra_args() -> None:
    agent = SylliptorAgent(extra_env={"SYLLIPTOR_EXTRA_ARGS": "--label 'unterminated"})

    with pytest.raises(RuntimeError, match="Invalid SYLLIPTOR_EXTRA_ARGS"):
        agent._extra_cli_args()


def test_harbor_adapter_requires_endpoint_and_key() -> None:
    missing_endpoint = SylliptorAgent(model_name="provider/model", extra_env={})
    with pytest.raises(RuntimeError, match="SYLLIPTOR_BASE_URL"):
        missing_endpoint._base_url()

    missing_key = SylliptorAgent(
        model_name="provider/model",
        extra_env={"SYLLIPTOR_BASE_URL": "https://example.invalid/v1"},
    )
    with pytest.raises(RuntimeError, match="SYLLIPTOR_API_KEY"):
        missing_key._container_env()


def test_harbor_run_command_quotes_instruction_and_extra_arguments() -> None:
    agent = SylliptorAgent(extra_env={"SYLLIPTOR_EXTRA_ARGS": "--label '$(touch /tmp/not-run)'"})
    instruction = "implement it; $(touch /tmp/also-not-run)"

    command = agent._build_run_command(
        instruction,
        model="provider/model",
        base_url="https://example.invalid/v1",
    )

    assert shlex.quote(instruction) in command
    assert shlex.quote("$(touch /tmp/not-run)") in command
    assert "SECRET-KEY" not in command
    assert "--api-key-env SYLLIPTOR_API_KEY" in command


def test_harbor_adapter_uses_the_repository_setup_script() -> None:
    agent = SylliptorAgent()

    assert Path(agent._host_setup_script_path()).samefile(
        Path(adapter_mod.__file__).with_name("setup.sh")
    )
    assert "chmod 1777" in agent._install_command()


def test_harbor_adapter_install_env_uses_wheel_without_credentials(tmp_path: Path) -> None:
    wheel = tmp_path / "sylliptor_agent_cli-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    agent = SylliptorAgent(
        extra_env={
            "SYLLIPTOR_WHEEL": str(wheel),
            "SYLLIPTOR_MODEL": "provider/model",
            "SYLLIPTOR_BASE_URL": "https://example.invalid/v1",
            "SYLLIPTOR_API_KEY": "test-only-key",
        }
    )

    install_env = agent._install_env()

    assert install_env["SYLLIPTOR_WHEEL"] == "/tmp/sylliptor-agent/" + wheel.name
    assert install_env["SYLLIPTOR_MODEL"] == "provider/model"
    assert install_env["SYLLIPTOR_BASE_URL"] == "https://example.invalid/v1"
    assert install_env["SYLLIPTOR_SETUP_LOG_DIR"] == "/logs/agent/setup"
    assert install_env["SYLLIPTOR_SETUP_ARTIFACT_DIR"] == "/logs/artifacts/setup"
    assert "SYLLIPTOR_API_KEY" not in install_env


def test_harbor_adapter_marks_infrastructure_exit_separately() -> None:
    agent = SylliptorAgent(
        model_name="provider/model",
        extra_env={
            "SYLLIPTOR_BASE_URL": "https://example.invalid/v1",
            "SYLLIPTOR_API_KEY": "test-only-key",
        },
    )
    calls = 0

    async def fake_exec_as_agent(
        _environment: object,
        *,
        command: str,
        env: dict[str, str],
        **_kwargs: object,
    ) -> None:
        nonlocal calls
        _ = command, env
        calls += 1
        if calls == 3:
            raise RuntimeError(
                f"Command failed (exit {INFRASTRUCTURE_FAILURE_EXIT_CODE}): provider outage"
            )

    agent.exec_as_agent = fake_exec_as_agent  # type: ignore[attr-defined,method-assign]
    context = SimpleNamespace(metadata={})

    with pytest.raises(RuntimeError, match="provider outage"):
        asyncio.run(agent.run("fix the bug", object(), context))

    assert context.metadata["sylliptor_exit_code"] == INFRASTRUCTURE_FAILURE_EXIT_CODE
    assert context.metadata["sylliptor_outcome"] == RunOutcome.INFRA_FAIL.value
