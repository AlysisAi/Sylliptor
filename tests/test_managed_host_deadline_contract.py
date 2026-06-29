from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sylliptor_agent_cli import agent_loop
from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.cli import app as sylliptor_app
from sylliptor_agent_cli.config import AppConfig, ConfigError


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "SYLLIPTOR_CONFIG_DIR": os.fspath(tmp_path / "cfg"),
        "SYLLIPTOR_DATA_DIR": os.fspath(tmp_path / "data"),
        "SYLLIPTOR_API_KEY": "",
        "OPENAI_API_KEY": "",
    }


class _FakeRunSession:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.turns: list[tuple[str, list[str] | None]] = []
        self.closed = False

    def run_turn(self, instruction: str, image_paths: list[str] | None = None) -> int:
        self.turns.append((instruction, image_paths))
        return self.exit_code

    def close(self) -> None:
        self.closed = True


def _run_agent_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cfg": AppConfig(model="test-model", stream=False),
        "root": tmp_path,
        "instruction": "do the task",
        "mode": "auto",
        "runtime_kind": "one_shot",
        "yes": True,
        "max_steps": 2,
        "no_log": True,
        "api_key_override": "override-key",
        "one_shot_execution": True,
        "enable_compaction": False,
        "verification_enabled": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_require_deadline_with_valid_cli_deadline_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_session = _FakeRunSession()

    def fake_create_session(**kwargs: Any) -> _FakeRunSession:
        captured.update(kwargs)
        return fake_session

    monkeypatch.setattr(agent_loop, "create_session", fake_create_session)

    code = agent_loop.run_agent(
        **_run_agent_kwargs(
            tmp_path,
            run_deadline_seconds=12.5,
            require_run_deadline=True,
        )
    )

    assert code == 0
    assert fake_session.closed is True
    deadline = captured["execution_deadline"]
    assert deadline.enabled is True
    assert deadline.configured_duration_seconds == 12.5
    assert str(deadline.source) == "explicit_cli"


@pytest.mark.parametrize(
    ("cfg", "env_value", "expected_seconds", "expected_source"),
    [
        (AppConfig(model="test-model", run_deadline_seconds=33.0), None, 33.0, "config"),
        (AppConfig(model="test-model", run_deadline_seconds=33.0), "44.0", 44.0, "environment"),
    ],
)
def test_require_deadline_accepts_environment_and_config_deadlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg: AppConfig,
    env_value: str | None,
    expected_seconds: float,
    expected_source: str,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_session(**kwargs: Any) -> _FakeRunSession:
        captured.update(kwargs)
        return _FakeRunSession()

    if env_value is None:
        monkeypatch.delenv("SYLLIPTOR_RUN_DEADLINE_SECONDS", raising=False)
    else:
        monkeypatch.setenv("SYLLIPTOR_RUN_DEADLINE_SECONDS", env_value)
    monkeypatch.setattr(agent_loop, "create_session", fake_create_session)

    code = agent_loop.run_agent(
        **_run_agent_kwargs(
            tmp_path,
            cfg=cfg,
            run_deadline_seconds=None,
            require_run_deadline=True,
        )
    )

    assert code == 0
    deadline = captured["execution_deadline"]
    assert deadline.enabled is True
    assert deadline.configured_duration_seconds == expected_seconds
    assert str(deadline.source) == expected_source


def test_require_deadline_fails_before_session_when_deadline_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_log = tmp_path / "diagnostics.jsonl"
    monkeypatch.delenv("SYLLIPTOR_RUN_DEADLINE_SECONDS", raising=False)

    def fail_create_session(**_kwargs: Any) -> _FakeRunSession:
        raise AssertionError("create_session should not be called")

    monkeypatch.setattr(agent_loop, "create_session", fail_create_session)

    with pytest.raises(ConfigError, match="requires a finite run deadline"):
        agent_loop.run_agent(
            **_run_agent_kwargs(
                tmp_path,
                require_run_deadline=True,
                crash_diagnostic_log_path=diagnostic_log,
            )
        )

    events = [
        json.loads(line)
        for line in diagnostic_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["event_type"] == "required_run_deadline_missing"
    assert events[-1]["payload"]["status"] == "blocked"
    assert events[-1]["payload"]["reason"] == "required_deadline_absent"
    assert events[-1]["payload"]["deadline_config_source"] == "absent"


def test_default_local_run_still_allows_absent_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.delenv("SYLLIPTOR_RUN_DEADLINE_SECONDS", raising=False)

    def fake_create_session(**kwargs: Any) -> _FakeRunSession:
        captured.update(kwargs)
        return _FakeRunSession()

    monkeypatch.setattr(agent_loop, "create_session", fake_create_session)

    code = agent_loop.run_agent(**_run_agent_kwargs(tmp_path))

    assert code == 0
    assert captured["execution_deadline"] is None


def test_run_agent_return_type_and_create_session_monkeypatch_remain_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = _FakeRunSession(exit_code=7)

    def fake_create_session(**_kwargs: Any) -> _FakeRunSession:
        return fake_session

    monkeypatch.setattr(agent_loop, "create_session", fake_create_session)

    code = agent_loop.run_agent(
        **_run_agent_kwargs(
            tmp_path,
            run_deadline_seconds=10,
            require_run_deadline=True,
            instruction="return-code probe",
        )
    )

    assert code == 7
    assert fake_session.turns == [("return-code probe", None)]
    assert fake_session.closed is True


def test_run_cli_preserves_dash_leading_instruction_after_separator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("demo\n", encoding="utf-8")
    instruction = "--starts-with-dash\nquote 'x' and shell $(echo nope)"
    captured: dict[str, Any] = {}

    def fake_run_agent(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_mod, "run_agent", fake_run_agent)

    result = runner.invoke(
        sylliptor_app,
        [
            "run",
            "--path",
            os.fspath(repo),
            "--model",
            "test-model",
            "--api-key",
            "k",
            "--no-log",
            "--yes",
            "--deadline-seconds",
            "12",
            "--require-deadline",
            "--",
            instruction,
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert captured["instruction"] == instruction
    assert captured["run_deadline_seconds"] == 12.0
    assert captured["require_run_deadline"] is True
