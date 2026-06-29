from __future__ import annotations

import json
import threading
from pathlib import Path

from sylliptor_agent_cli.crash_diagnostics import (
    CRASH_DIAGNOSTIC_SCHEMA_VERSION,
    CrashDiagnosticLogger,
    build_crash_diagnostic_logger,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_diagnostic_logger_disabled_by_default_creates_no_file(tmp_path: Path) -> None:
    logger = CrashDiagnosticLogger.disabled()

    logger.event("run_started", {"status": "started"}, durable=True)

    assert list(tmp_path.iterdir()) == []


def test_diagnostic_logger_writes_incremental_jsonl_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "diag" / "events.jsonl"
    logger = build_crash_diagnostic_logger(
        path=path,
        run_id="run-1",
        session_id="session-1",
        runtime_kind="one_shot",
    )

    logger.event("run_started", {"status": "started"}, durable=True)

    events = _read_jsonl(path)
    assert len(events) == 1
    event = events[0]
    assert event["schema_version"] == CRASH_DIAGNOSTIC_SCHEMA_VERSION
    assert event["event_type"] == "run_started"
    assert event["run_id"] == "run-1"
    assert event["session_id"] == "session-1"
    assert event["payload"] == {"runtime_kind": "one_shot", "status": "started"}


def test_diagnostic_logger_privacy_allowlist_excludes_sensitive_fields(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sentinel = "SECRET_TOKEN_DO_NOT_WRITE"
    logger = build_crash_diagnostic_logger(
        path=path,
        run_id="run-privacy",
        session_id="session-privacy",
    )

    logger.event(
        "tool_started",
        {
            "tool_name": "shell_run",
            "arguments": {"cmd": f"echo {sentinel}"},
            "stdout": sentinel,
            "prompt": sentinel,
            "authorization": f"Bearer {sentinel}",
            "source": f"print({sentinel!r})",
        },
    )

    contents = path.read_text(encoding="utf-8")
    assert sentinel not in contents
    events = _read_jsonl(path)
    assert events[0]["payload"] == {"tool_name": "shell_run"}


def test_diagnostic_logger_keeps_deadline_finalization_fields(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = build_crash_diagnostic_logger(
        path=path,
        run_id="run-deadline",
        session_id="session-deadline",
    )

    logger.event(
        "deadline_operation_blocked",
        {
            "operation": "subagent",
            "deadline": {
                "enabled": True,
                "source": "explicit_cli",
                "phase": "finalization_window",
                "finalization_reserve_seconds": 5.0,
                "normal_work_remaining_seconds": 0.0,
                "finalization_reason": "reserve_reached",
                "duration_observations": {"main_llm": {"count": 1, "average_seconds": 4.0}},
                "secret": "drop-me",
            },
            "deadline_start_decision": {
                "operation": "subagent",
                "allowed": False,
                "reason": "finalization_disallows_operation",
                "phase": "finalization_window",
            },
        },
    )

    payload = _read_jsonl(path)[0]["payload"]
    assert payload["deadline"]["phase"] == "finalization_window"
    assert payload["deadline"]["source"] == "explicit_cli"
    assert payload["deadline"]["duration_observations"]["main_llm"]["count"] == 1
    assert "secret" not in payload["deadline"]
    assert payload["deadline_start_decision"]["reason"] == "finalization_disallows_operation"


def test_diagnostic_logger_failure_is_isolated_for_unwritable_path(tmp_path: Path) -> None:
    logger = build_crash_diagnostic_logger(
        path=tmp_path,
        run_id="run-dir",
        session_id="session-dir",
    )

    logger.event("terminal_error", {"failure_category": "io"}, durable=True)

    assert tmp_path.is_dir()


def test_diagnostic_logger_concurrent_writes_are_complete_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = build_crash_diagnostic_logger(
        path=path,
        run_id="run-concurrent",
        session_id="session-concurrent",
    )

    def write_events(offset: int) -> None:
        for step in range(offset, offset + 10):
            logger.event("step_started", {"step": step})

    threads = [threading.Thread(target=write_events, args=(idx * 10,)) for idx in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = _read_jsonl(path)
    assert len(events) == 40
    assert sorted(event["payload"]["step"] for event in events) == list(range(40))
